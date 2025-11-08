import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
import random
from collections import deque
from sentence_transformers import SentenceTransformer
from .identity_module import Identity
from .adaptive_event_detector import AdaptiveEventDetector
from .memory_controller import MemoryController
from .ground_truth_simulator import GroundTruthSimulator
from .replay_buffer import ExperienceReplayBuffer
from . import config

class MemoryAI:
    """
    The main class for the Memory and Identity AI stack.

    This class integrates all the components of the system, including the
    identity module, event detector, memory controller, and ground truth
    simulator. It orchestrates the process of handling interactions,
    detecting events, updating memory, and training the memory update model.
    """
    def __init__(self, user_id, initial_identity_properties, state_filepath=None, training_log_path=None):
        """
        Initializes the MemoryAI instance.

        Args:
            user_id (str): The unique identifier for the user.
            initial_identity_properties (dict): A dictionary of initial
                properties for the user's identity.
            state_filepath (str, optional): The path to the file where the
                agent's state is stored. Defaults to None.
            training_log_path (str, optional): The path to the file where
                training data is logged. Defaults to None.
        """
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
        self.replay_buffer = ExperienceReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
        self.replay_ratio = config.REPLAY_SAMPLE_RATIO

        self.optimizer = optim.AdamW(self.memory_controller.f_theta.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=config.PATIENCE)
        self.loss_function = nn.MSELoss()

        self.best_loss = float('inf')
        self.patience_counter = 0

        self.state_filepath = state_filepath
        if self.state_filepath:
            self.memory_controller.load_state(self.state_filepath)

        self.training_log_path = training_log_path
        self.replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)

        self.last_interaction = None
        self.last_explanation_data = None

    def save_state(self):
        """
        Saves the agent's state to the file specified by `state_filepath`.
        """
        if self.state_filepath:
            self.memory_controller.save_state(self.state_filepath)

    def _prepare_input_tensors(self, memory_state, identity_tensor, event_data):
        """
        Creates an event tensor and returns a dictionary of input tensors.

        Args:
            memory_state (torch.Tensor): The current memory state.
            identity_tensor (torch.Tensor): The user's identity tensor.
            event_data (dict): The event data.

        Returns:
            dict: A dictionary of input tensors for the memory update model.
        """
        event_content = event_data.get("content", "")
        event_tensor = self.embedding_model.encode(event_content, convert_to_tensor=True).to(self.device).unsqueeze(0).clone()
        return {
            "memory_state": memory_state.to(self.device),
            "identity_tensor": identity_tensor.to(self.device),
            "event_tensor": event_tensor
        }

    def log_training_data(self, input_tensors, target_delta_m):
        """Logs the training data to a file."""
        data = {
            "inputs": {k: v.cpu().tolist() for k, v in input_tensors.items()},
            "target": target_delta_m.cpu().tolist()
        }
        self.replay_buffer.append(data)

        if self.training_log_path:
            with open(self.training_log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')

    def process_interaction(self, interaction_data):
        """
        Processes an interaction, detects events, updates memory, and logs for training.

        Args:
            interaction_data (dict): The interaction data.

        Returns:
            dict: A dictionary of input tensors if an event is detected,
                  otherwise None.
        """
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

        self.save_state()

        return input_tensors

    def train_on_batch(self, batch_data):
        """Performs a training step on a batch of logged data."""
        if len(self.replay_buffer) > len(batch_data):
            replay_sample_size = min(len(self.replay_buffer), len(batch_data) * 2)
            replay_samples = random.sample(self.replay_buffer, replay_sample_size)
            batch_data.extend(replay_samples)

        total_loss = 0.0
        for data in batch_data:
            self.replay_buffer.add(data)

        # Sample replay experiences
        replay_size = int(len(batch_data) * self.replay_ratio)
        replay_samples = self.replay_buffer.sample(replay_size)

        # Combine and shuffle
        combined_batch = batch_data + replay_samples
        random.shuffle(combined_batch)

        logging.info(f"Training: {len(batch_data)} new + {len(replay_samples)} replay")

        # Train on combined batch (existing code)
        total_loss = 0.0
        for data in combined_batch:
            input_tensors = {k: torch.tensor(v).to(self.device)
                             for k, v in data['inputs'].items()}
            target_output = torch.tensor(data['target']).to(self.device)

            self.optimizer.zero_grad()
            output = self.memory_controller.predict_delta_m(**input_tensors)
            loss = self.loss_function(output, target_output)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.memory_controller.f_theta.parameters(),
                config.MAX_GRAD_NORM
            )
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(combined_batch)
        self.scheduler.step(avg_loss)

        logging.info(f"Training complete: loss={avg_loss:.6f}")
        return avg_loss
