import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
from sentence_transformers import SentenceTransformer
from .identity_module import Identity
from .adaptive_event_detector import AdaptiveEventDetector
from .memory_controller import MemoryController
from .ground_truth_simulator import GroundTruthSimulator
from . import config

class MemoryAI:
    """Integrates components and orchestrates the memory update and learning process."""
    def __init__(self, user_id, initial_identity_properties, state_filepath=None, training_log_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.identity = Identity(user_id, initial_identity_properties, self.embedding_model)
        self.event_detector = AdaptiveEventDetector()
        self.ground_truth_simulator = GroundTruthSimulator(self.embedding_model, self.device)

        self.memory_controller = MemoryController(
            memory_size=config.MEMORY_SIZE,
            identity_size=config.IDENTITY_EMBEDDING_SIZE,
            event_size=config.EVENT_EMBEDDING_SIZE
        )
        self.memory_controller.f_theta.to(self.device)

        self.optimizer = optim.AdamW(self.memory_controller.f_theta.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=config.PATIENCE)
        self.loss_function = nn.MSELoss()

        self.best_loss = float('inf')
        self.patience_counter = 0

        self.state_filepath = state_filepath
        if self.state_filepath:
            self.memory_controller.load_state(self.state_filepath)

        self.training_log_path = training_log_path

    def save_state(self):
        """Saves the agent's state."""
        if self.state_filepath:
            self.memory_controller.save_state(self.state_filepath)

    def _prepare_input_tensors(self, memory_state, identity_tensor, event_data):
        """Creates an event tensor and returns a dictionary of input tensors."""
        event_content = event_data.get("content", "")
        event_tensor = self.embedding_model.encode(event_content, convert_to_tensor=True).to(self.device).unsqueeze(0).clone()
        return {
            "memory_state": memory_state.to(self.device),
            "identity_tensor": identity_tensor.to(self.device),
            "event_tensor": event_tensor
        }

    def log_training_data(self, input_tensors, target_delta_m):
        """Logs the training data to a file."""
        if not self.training_log_path:
            return

        data = {
            "inputs": {k: v.tolist() for k, v in input_tensors.items()},
            "target": target_delta_m.tolist()
        }
        with open(self.training_log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def process_interaction(self, interaction_data):
        """Processes an interaction, detects events, updates memory, and logs for training."""
        event = self.event_detector.detect(interaction_data)
        if not event:
            return None

        logging.info(f"Event detected: {event['event_type']} - '{interaction_data['content']}' (Threshold: {self.event_detector.threshold:.2f})")

        old_identity_embedding = self.identity.get_properties_tensor()

        if interaction_data.get('type') == 'identity_update':
            self.identity.update_properties(interaction_data)

        new_identity_embedding = self.identity.get_properties_tensor()

        memory_state = self.memory_controller.get_state()
        input_tensors = self._prepare_input_tensors(memory_state, old_identity_embedding, event['data'])

        predicted_delta_m = self.memory_controller.predict_delta_m(**input_tensors)
        self.memory_controller.update(predicted_delta_m.detach())

        target_delta_m = self.ground_truth_simulator.get_target_delta_m(event['data'], old_identity_embedding, new_identity_embedding)

        self.log_training_data(input_tensors, target_delta_m)

        return input_tensors

    def train_on_batch(self, batch_data):
        """Performs a training step on a batch of logged data."""
        total_loss = 0.0
        for data in batch_data:
            input_tensors = {k: torch.tensor(v).to(self.device) for k, v in data['inputs'].items()}
            target_output = torch.tensor(data['target']).to(self.device)

            self.optimizer.zero_grad()
            output = self.memory_controller.predict_delta_m(**input_tensors)
            loss = self.loss_function(output, target_output)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.memory_controller.f_theta.parameters(), config.MAX_GRAD_NORM)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batch_data)
        self.scheduler.step(avg_loss)
        logging.info(f"  Batch training complete. Average loss: {avg_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        return avg_loss
