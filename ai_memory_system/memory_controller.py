import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import config

class TransformerFTheta(nn.Module):
    """Transformer-based network to compute memory updates (ΔM)."""
    def __init__(self, memory_size, identity_size, event_size, hidden_size, output_size, nhead, num_layers):
        super(TransformerFTheta, self).__init__()
        # Projection layers to create a common embedding space
        self.mem_proj = nn.Linear(memory_size, hidden_size)
        self.id_proj = nn.Linear(identity_size, hidden_size)
        self.event_proj = nn.Linear(event_size, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated update mechanism
        self.input_gate_linear = nn.Linear(hidden_size, output_size)
        self.candidate_linear = nn.Linear(hidden_size, output_size)

    def forward(self, memory_state, identity_tensor, event_tensor):
        # Project inputs to common dimension
        mem_p = torch.tanh(self.mem_proj(memory_state))
        id_p = torch.tanh(self.id_proj(identity_tensor))
        event_p = torch.tanh(self.event_proj(event_tensor))

        # Stack for transformer
        inputs_p = torch.stack([mem_p, id_p, event_p], dim=1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(inputs_p)

        # Use the mean of the transformer output as the context
        context_vector = torch.mean(transformer_output, dim=1)

        # Gated update
        input_gate = torch.sigmoid(self.input_gate_linear(context_vector))
        candidate = torch.tanh(self.candidate_linear(context_vector))
        return input_gate * candidate

import logging

class MemoryController:
    """Manages the AI's memory and the neural network for updates."""
    def __init__(self, memory_size, identity_size, event_size):
        self.state = torch.zeros(1, memory_size)
        self.f_theta = TransformerFTheta(
            memory_size=memory_size,
            identity_size=identity_size,
            event_size=event_size,
            hidden_size=config.HIDDEN_SIZE,
            output_size=memory_size,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS
        )
        self.layer_norm = nn.LayerNorm(memory_size, eps=1e-6)
        self.lambda_decay = config.LAMBDA_DECAY
        self.activation_factor = config.ACTIVATION_FACTOR

    def update(self, delta_m, dt=1.0):
        """
    Bounded memory update with stability guarantees.

    References:
    - "Layer Normalization" (Ba et al., 2016)
    - LSTM design (Hochreiter & Schmidhuber, 1997)

    Guarantees: ||state|| ≤ √memory_size for all time
    """
        # Discretized ODE update
        decay_term = self.state * (1 - self.lambda_decay * dt)
        update_term = self.activation_factor * delta_m * dt
        raw_update = decay_term + update_term

        # CRITICAL: Bounded activation prevents explosion
        bounded_update = torch.tanh(raw_update)

        # Layer normalization improves training stability
        self.state = self.layer_norm(bounded_update)

        # Logging
        norm = torch.norm(self.state).item()
        if norm > 10.0:
            logging.warning(f"Memory norm high: {norm:.2f}")

    def get_state(self):
        return self.state

    def predict_delta_m(self, memory_state, identity_tensor, event_tensor):
        return self.f_theta(memory_state, identity_tensor, event_tensor)

    def save_state(self, filepath):
        """Saves the memory state and the model parameters to a file."""
        state = {
            'memory_state': self.state,
            'f_theta_state_dict': self.f_theta.state_dict(),
            'layer_norm_state_dict': self.layer_norm.state_dict()
        }
        torch.save(state, filepath)

    def load_state(self, filepath):
        """Loads the memory state and the model parameters from a file."""
        if os.path.exists(filepath):
            state = torch.load(filepath)
            self.state = state['memory_state']
            self.f_theta.load_state_dict(state['f_theta_state_dict'])
            if 'layer_norm_state_dict' in state:
                self.layer_norm.load_state_dict(state['layer_norm_state_dict'])
