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
        self.identity_embedding = self._create_embedding()

    def _hash_properties(self):
        """Hashes the user's properties."""
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return hashlib.sha256(prop_str.encode()).hexdigest()

    def _create_embedding(self):
        """Converts identity properties to a fixed-size tensor with differential privacy."""
        hashed_props = self._hash_properties()
        embedding = self.embedding_model.encode(hashed_props, convert_to_tensor=True).unsqueeze(0)

        # Apply differential privacy
        laplace = Laplace(epsilon=self.epsilon, sensitivity=1.0)
        noisy_embedding = torch.tensor([laplace.randomise(x.item()) for x in embedding.flatten()]).reshape(embedding.shape)

        return noisy_embedding.float()

    def get_properties_tensor(self):
        """Returns the identity embedding."""
        return self.identity_embedding

    def update_properties(self, new_properties, alpha=0.1):
        """Updates the user's properties and the identity embedding."""
        self.properties.update(new_properties)
        new_embedding = self._create_embedding()
        self.identity_embedding = (1 - alpha) * self.identity_embedding + alpha * new_embedding
