import torch
import torch.nn as nn

class FTheta(nn.Module):
    """Neural network to compute memory updates (ΔM)."""
    def __init__(self, input_size, hidden_size, output_size):
        super(FTheta, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class MemoryController:
    """Manages the AI's memory and the neural network for updates."""
    def __init__(self, memory_size=128, input_size=256, hidden_size=256, lambda_decay=0.01, activation_factor=1.0):
        self.state = torch.zeros(1, memory_size)
        self.f_theta = FTheta(input_size=input_size, hidden_size=hidden_size, output_size=memory_size)
        self.lambda_decay = lambda_decay
        self.activation_factor = activation_factor

    def update(self, delta_m, dt=1.0):
        """Discretized memory update equation."""
        self.state = self.state * (1 - self.lambda_decay * dt) + self.activation_factor * delta_m * dt

    def get_state(self):
        return self.state

    def predict_delta_m(self, combined_input):
        return self.f_theta(combined_input)
