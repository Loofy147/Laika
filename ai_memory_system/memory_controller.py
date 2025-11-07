import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFTheta(nn.Module):
    """Transformer-based network to compute memory updates (ΔM)."""
    def __init__(self, memory_size, identity_size, event_size, hidden_size, output_size, nhead=4, num_layers=2):
        super(TransformerFTheta, self).__init__()
        # Projection layers to create a common embedding space
        self.mem_proj = nn.Linear(memory_size, hidden_size)
        self.id_proj = nn.Linear(identity_size, hidden_size)
        self.event_proj = nn.Linear(event_size, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated update mechanism
        self.input_gate_linear = nn.Linear(hidden_size, output_size)
        self.candidate_linear = nn.Linear(hidden_size, output_size)

    def forward(self, memory_state, identity_tensor, event_tensor):
        # Project inputs to common dimension
        mem_p = torch.tanh(self.mem_proj(memory_state))
        id_p = torch.tanh(self.id_proj(identity_tensor))
        event_p = torch.tanh(self.event_proj(event_tensor))

        # Stack for transformer and add batch dimension
        inputs_p = torch.stack([mem_p.squeeze(0), id_p.squeeze(0), event_p.squeeze(0)], dim=0).unsqueeze(1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(inputs_p)

        # Use the mean of the transformer output as the context
        context_vector = torch.mean(transformer_output.squeeze(1), dim=0).unsqueeze(0)

        # Gated update
        input_gate = torch.sigmoid(self.input_gate_linear(context_vector))
        candidate = torch.tanh(self.candidate_linear(context_vector))
        return input_gate * candidate

class MemoryController:
    """Manages the AI's memory and the neural network for updates."""
    def __init__(self, memory_size, identity_size, event_size, hidden_size=256, lambda_decay=0.01, activation_factor=1.0):
        self.state = torch.zeros(1, memory_size)
        self.f_theta = TransformerFTheta(
            memory_size=memory_size,
            identity_size=identity_size,
            event_size=event_size,
            hidden_size=hidden_size,
            output_size=memory_size
        )
        self.lambda_decay = lambda_decay
        self.activation_factor = activation_factor

    def update(self, delta_m, dt=1.0):
        """Discretized memory update equation."""
        self.state = self.state * (1 - self.lambda_decay * dt) + self.activation_factor * delta_m * dt

    def get_state(self):
        return self.state

    def predict_delta_m(self, memory_state, identity_tensor, event_tensor):
        return self.f_theta(memory_state, identity_tensor, event_tensor)
