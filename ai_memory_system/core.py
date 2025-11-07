import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from SALib.sample import saltelli
from SALib.analyze import sobol
from .identity_module import Identity
from .adaptive_event_detector import AdaptiveEventDetector
from .memory_controller import MemoryController

class MemoryAI:
    """Integrates components and orchestrates the memory update and learning process."""
    def __init__(self, user_id, initial_identity_properties, memory_size=128, identity_embedding_size=384, event_embedding_size=384, patience=5, min_delta=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.identity = Identity(user_id, initial_identity_properties, self.embedding_model)
        self.event_detector = AdaptiveEventDetector(initial_threshold=0.6)

        self.input_size = memory_size + identity_embedding_size + event_embedding_size
        self.memory_controller = MemoryController(memory_size=memory_size, input_size=self.input_size, hidden_size=512)
        self.memory_controller.f_theta.to(self.device)

        self.optimizer = optim.AdamW(self.memory_controller.f_theta.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=patience)
        self.loss_function = nn.MSELoss()
        self.max_grad_norm = 1.0

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0

    def _prepare_input_tensor(self, memory_state, identity_tensor, event_data):
        """Creates an event tensor and combines inputs."""
        event_content = event_data.get("content", "")
        event_tensor = self.embedding_model.encode(event_content, convert_to_tensor=True).to(self.device).unsqueeze(0)
        return torch.cat((memory_state.to(self.device), identity_tensor.to(self.device), event_tensor), dim=1)

    def _get_simulated_target_delta_m(self, event_data):
        """Creates a simulated 'ground truth' memory update for learning."""
        content = event_data.get("content", "")
        significance = event_data.get("significance", 0)

        target_embedding = self.embedding_model.encode(content, convert_to_tensor=True).to(self.device)

        memory_dim = self.memory_controller.state.shape[1]
        if target_embedding.shape[0] < memory_dim:
            padding = torch.zeros(memory_dim - target_embedding.shape[0], device=self.device)
            target_embedding = torch.cat((target_embedding, padding))
        elif target_embedding.shape[0] > memory_dim:
            target_embedding = target_embedding[:memory_dim]

        return significance * target_embedding.unsqueeze(0)

    def process_interaction(self, interaction_data):
        """Processes an interaction, detects events, and updates memory."""
        event = self.event_detector.detect(interaction_data)
        if event:
            logging.info(f"Event detected: {event['event_type']} - '{interaction_data['content']}' (Threshold: {self.event_detector.threshold:.2f})")
            loss = self.update_memory_and_learn(event['data'])
            return loss
        return None

    def update_memory_and_learn(self, event_data, dt=1.0):
        """Updates memory based on an event and performs a learning step."""
        memory_state = self.memory_controller.get_state()
        identity_tensor = self.identity.get_properties_tensor()

        combined_input = self._prepare_input_tensor(memory_state, identity_tensor, event_data)
        predicted_delta_m = self.memory_controller.predict_delta_m(combined_input)

        target_delta_m = self._get_simulated_target_delta_m(event_data)
        self.memory_controller.update(predicted_delta_m.detach(), dt=dt)

        loss = self.train_f_theta(combined_input, target_delta_m)
        return loss

    def train_f_theta(self, input_tensor, target_output):
        """Performs a single training step for the f_theta network."""
        self.optimizer.zero_grad()
        output = self.memory_controller.predict_delta_m(input_tensor)
        loss = self.loss_function(output, target_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.memory_controller.f_theta.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step(loss.item())

        logging.info(f"  Training loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        if self.best_loss - loss.item() > self.min_delta:
            self.best_loss = loss.item()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logging.info("Convergence detected. Stopping training.")
                return None

        return loss.item()

    def analyze_sensitivity(self, param_ranges, interactions):
        """Performs Sobol sensitivity analysis."""
        problem = {
            'num_vars': len(param_ranges),
            'names': list(param_ranges.keys()),
            'bounds': list(param_ranges.values())
        }
        param_values = saltelli.sample(problem, 1024)

        Y = np.zeros([param_values.shape[0]])
        for i, X in enumerate(param_values):
            self.memory_controller.lambda_decay = X[0]
            self.memory_controller.activation_factor = X[1]
            for interaction in interactions:
                self.process_interaction(interaction)
            Y[i] = torch.norm(self.memory_controller.get_state()).item()

        Si = sobol.analyze(problem, Y)
        return Si

    def run_monte_carlo_simulation(self, interactions, n_simulations=100):
        """Runs a Monte Carlo simulation to quantify uncertainty."""
        final_memory_norms = []
        for _ in range(n_simulations):
            for interaction in interactions:
                interaction['significance'] += np.random.normal(0, 0.1)
                self.process_interaction(interaction)
            final_memory_norms.append(torch.norm(self.memory_controller.get_state()).item())
        return final_memory_norms
