import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
from .identity_module import Identity
from .event_detection import EventDetector

class Memory:
    """Manages the AI's memory."""
    def __init__(self, memory_size=128):
        self.state = torch.zeros(1, memory_size)

    def update(self, delta_m):
        self.state += delta_m

    def get_state(self):
        return self.state

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

class MemoryAI:
    """Integrates components and orchestrates the memory update and learning process."""
    def __init__(self, user_id, initial_identity_properties, memory_size=128, identity_embedding_size=32, event_embedding_size=96):
        self.memory = Memory(memory_size)
        self.identity = Identity(user_id, initial_identity_properties, identity_embedding_size)
        self.event_detector = EventDetector(event_threshold=0.6)

        self.input_size = memory_size + identity_embedding_size + event_embedding_size
        self.f_theta = FTheta(input_size=self.input_size, hidden_size=256, output_size=memory_size)
        self.optimizer = optim.AdamW(self.f_theta.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=5)
        self.loss_function = nn.MSELoss()
        self.max_grad_norm = 1.0

    def _prepare_input_tensor(self, memory_state, identity_tensor, event_data):
        """Hashes event data to create a reproducible event tensor and combines inputs."""
        event_str = str(event_data)
        seed = int(hashlib.sha256(event_str.encode('utf-8')).hexdigest(), 16) % (10**8)
        torch.manual_seed(seed)
        event_tensor = torch.randn(1, self.f_theta.network[0].in_features - memory_state.shape[1] - identity_tensor.shape[1])
        return torch.cat((memory_state, identity_tensor, event_tensor), dim=1)

    def _get_simulated_target_delta_m(self, event_data):
        """Creates a simulated 'ground truth' memory update for learning."""
        content = event_data.get("content", "")
        significance = event_data.get("significance", 0)

        seed = int(hashlib.sha256(content.encode('utf-8')).hexdigest(), 16) % (10**8)
        torch.manual_seed(seed)

        base_update = torch.randn(1, self.memory.state.shape[1])
        return significance * base_update

    def process_interaction(self, interaction_data):
        """Processes an interaction, detects events, and updates memory."""
        event = self.event_detector.detect(interaction_data)
        if event:
            print(f"Event detected: {event['event_type']} - '{interaction_data['content']}'")
            loss = self.update_memory_and_learn(event['data'])
            return loss
        return None

    def update_memory_and_learn(self, event_data):
        """Updates memory based on an event and performs a learning step."""
        memory_state = self.memory.get_state()
        identity_tensor = self.identity.get_properties_tensor()

        combined_input = self._prepare_input_tensor(memory_state, identity_tensor, event_data)

        predicted_delta_m = self.f_theta(combined_input)

        target_delta_m = self._get_simulated_target_delta_m(event_data)
        self.memory.update(predicted_delta_m.detach())

        loss = self.train_f_theta(combined_input, target_delta_m)
        return loss

    def train_f_theta(self, input_tensor, target_output):
        """Performs a single training step for the f_theta network."""
        self.optimizer.zero_grad()
        output = self.f_theta(input_tensor)
        loss = self.loss_function(output, target_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.f_theta.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.scheduler.step(loss.item())

        print(f"  Training loss: {loss.item():.6f}")
        return loss.item()
