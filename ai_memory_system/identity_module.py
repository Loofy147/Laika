import torch
import hashlib
from sentence_transformers import SentenceTransformer
from diffprivlib.mechanisms import Laplace
from . import config

class Identity:
    """
    Represents the user's identity and handles property encoding.

    This class manages the user's properties, converts them into a
    privacy-preserving embedding using a combination of hashing, a sentence
    transformer model, and differential privacy.
    """
    def __init__(self, user_id, initial_properties, embedding_model):
        """
        Initializes the Identity instance.

        Args:
            user_id (str): The unique identifier for the user.
            initial_properties (dict): A dictionary of initial properties
                for the user's identity.
            embedding_model: The sentence transformer model to use for
                creating embeddings.
        """
        self.user_id = user_id
        self.properties = initial_properties
        self.embedding_model = embedding_model
        self.epsilon = config.EPSILON
        self.identity_embedding = self._create_embedding()

    def _hash_properties(self):
        """
        Hashes the user's properties to create a unique and privacy-preserving
        representation.

        Returns:
            str: The SHA-256 hash of the user's properties.
        """
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return hashlib.sha256(prop_str.encode()).hexdigest()

    def _create_embedding(self):
        """
        Converts identity properties to a fixed-size tensor with differential
        privacy.

        This method first hashes the properties, then creates an embedding of
        the hash using the sentence transformer model. Finally, it adds
        Laplace noise to the embedding to provide differential privacy.

        Returns:
            torch.Tensor: The privacy-preserving identity embedding.
        """
        hashed_props = self._hash_properties()
        embedding = self.embedding_model.encode(hashed_props, convert_to_tensor=True).unsqueeze(0)

        # Dynamically calculate sensitivity
        sensitivity = torch.norm(embedding).item()

        # Apply differential privacy
        laplace = Laplace(epsilon=self.epsilon, sensitivity=sensitivity)
        noisy_embedding = torch.tensor([laplace.randomise(x.item()) for x in embedding.flatten()]).reshape(embedding.shape)

        return noisy_embedding.float()

    def get_properties_tensor(self):
        """
        Returns the identity embedding.

        Returns:
            torch.Tensor: The current identity embedding.
        """
        return self.identity_embedding

    def update_properties(self, new_properties):
        """
        Updates the user's properties and the identity embedding.

        The new embedding is combined with the old embedding using an
        exponential moving average to provide a smooth transition.

        Args:
            new_properties (dict): A dictionary of new properties to add or
                update.
        """
        self.properties.update(new_properties)
        new_embedding = self._create_embedding()
        self.identity_embedding = (1 - config.ALPHA) * self.identity_embedding + config.ALPHA * new_embedding
