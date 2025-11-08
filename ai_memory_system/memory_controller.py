import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import config

class TransformerFTheta(nn.Module):
    """
    A Transformer-based model for computing memory updates (ΔM).

    This model uses a Transformer encoder to process the memory state,
    identity, and event tensors as a sequence. The output of the
    transformer is then used to generate a gated memory update.
    """
    def __init__(self, memory_size, identity_size, event_size, hidden_size, output_size, nhead, num_layers):
        """
        Initializes the TransformerFTheta model.

        Args:
            memory_size (int): The size of the memory state vector.
            identity_size (int): The size of the identity embedding.
            event_size (int): The size of the event embedding.
            hidden_size (int): The hidden size for the projection layers and
                the Transformer encoder.
            output_size (int): The size of the output vector (ΔM).
            nhead (int): The number of attention heads in the Transformer
                encoder.
            num_layers (int): The number of layers in the Transformer encoder.
        """
        super(TransformerFTheta, self).__init__()
        # Projection layers to create a common embedding space
        self.mem_proj = nn.Linear(memory_size, hidden_size)
        self.id_proj = nn.Linear(identity_size, hidden_size)
        self.event_proj = nn.Linear(event_size, hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(hidden_size)

        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Gated update mechanism
        self.input_gate_linear = nn.Linear(hidden_size, output_size)
        self.candidate_linear = nn.Linear(hidden_size, output_size)

    def forward(self, memory_state, identity_tensor, event_tensor):
        """
        Performs the forward pass of the model.

        Args:
            memory_state (torch.Tensor): The current memory state.
            identity_tensor (torch.Tensor): The user's identity tensor.
            event_tensor (torch.Tensor): The event tensor.

        Returns:
            torch.Tensor: The predicted memory update (ΔM).
        """
        # Project inputs to common dimension
        mem_p = self.ln1(torch.tanh(self.mem_proj(memory_state)))
        id_p = self.ln2(torch.tanh(self.id_proj(identity_tensor)))
        event_p = self.ln3(torch.tanh(self.event_proj(event_tensor)))

        # Stack for transformer
        inputs_p = torch.stack([mem_p, id_p, event_p], dim=1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(inputs_p)
        transformer_output = self.ln_out(transformer_output)

        # Use the mean of the transformer output as the context
        context_vector = torch.mean(transformer_output, dim=1)

        # Apply LayerNorm
        context_vector = self.layer_norm(context_vector)

        # Gated update
        input_gate = torch.sigmoid(self.input_gate_linear(context_vector))
        candidate = torch.tanh(self.candidate_linear(context_vector))
        return torch.tanh(input_gate * candidate)

import logging

class MemoryController:
    """
    Manages the AI's memory and the neural network for updates.

    This class is responsible for maintaining the memory state, predicting
    memory updates using the TransformerFTheta model, and applying the
    updates to the memory state.
    """
    def __init__(self, memory_size, identity_size, event_size):
        """
        Initializes the MemoryController.

        Args:
            memory_size (int): The size of the memory state vector.
            identity_size (int): The size of the identity embedding.
            event_size (int): The size of the event embedding.
        """
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
        Updates the memory state using a discretized differential equation.

        The equation is: M(t+dt) = M(t) * (1 - λ*dt) + a * ΔM * dt

        Args:
            delta_m (torch.Tensor): The predicted memory update.
            dt (float, optional): The time step. Defaults to 1.0.
        """
        self.state = self.state * (1 - self.lambda_decay * dt) + self.activation_factor * delta_m * dt
        self.state = F.normalize(self.state, p=2, dim=1)

        # Clip the memory state to a maximum norm
        max_norm = torch.sqrt(torch.tensor(self.state.shape[1]))
        current_norm = torch.norm(self.state)
        if current_norm > max_norm:
            self.state = self.state * (max_norm / current_norm)

    def get_state(self):
        """
        Returns the current memory state.

        Returns:
            torch.Tensor: The current memory state.
        """
        return self.state

    def predict_delta_m(self, memory_state, identity_tensor, event_tensor):
        """
        Predicts the memory update (ΔM) using the TransformerFTheta model.

        Args:
            memory_state (torch.Tensor): The current memory state.
            identity_tensor (torch.Tensor): The user's identity tensor.
            event_tensor (torch.Tensor): The event tensor.

        Returns:
            torch.Tensor: The predicted memory update (ΔM).
        """
        return self.f_theta(memory_state, identity_tensor, event_tensor)
