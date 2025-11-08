import random
from collections import deque

class ExperienceReplayBuffer:
    """A simple replay buffer to store and sample experiences."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """Add a new experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
