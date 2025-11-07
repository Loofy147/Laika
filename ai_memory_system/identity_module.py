import torch
import hashlib

class Identity:
    """Represents the user's identity and handles property encoding."""
    def __init__(self, user_id, initial_properties, embedding_size=32):
        self.user_id = user_id
        self.properties = initial_properties
        self.embedding_size = embedding_size

    def get_properties_tensor(self):
        """Converts identity properties to a fixed-size tensor."""
        prop_str = str(self.properties)
        seed = int(hashlib.sha256(prop_str.encode('utf-8')).hexdigest(), 16) % (10**8)
        torch.manual_seed(seed)
        return torch.randn(1, self.embedding_size)
