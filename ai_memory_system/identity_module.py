import torch
import hashlib
from sentence_transformers import SentenceTransformer
from diffprivlib.mechanisms import Laplace

class Identity:
    """Represents the user's identity and handles property encoding."""
    def __init__(self, user_id, initial_properties, embedding_model, epsilon=1.0):
        self.user_id = user_id
        self.properties = initial_properties
        self.embedding_model = embedding_model
        self.epsilon = epsilon

    def _hash_properties(self):
        """Hashes the user's properties."""
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return hashlib.sha256(prop_str.encode()).hexdigest()

    def get_properties_tensor(self):
        """Converts identity properties to a fixed-size tensor with differential privacy."""
        hashed_props = self._hash_properties()
        embedding = self.embedding_model.encode(hashed_props, convert_to_tensor=True).unsqueeze(0)

        # Apply differential privacy
        laplace = Laplace(epsilon=self.epsilon, sensitivity=1.0)
        noisy_embedding = torch.tensor([laplace.randomise(x.item()) for x in embedding.flatten()]).reshape(embedding.shape)

        return noisy_embedding.float()
