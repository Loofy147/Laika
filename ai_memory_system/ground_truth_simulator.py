import torch
from . import config

class GroundTruthSimulator:
    """
    Generates a simulated 'ground truth' for memory updates.

    This class creates a target memory update (ΔM) based on the principle that
    the memory should reflect the change in the user's identity embedding.
    The significance of the event is used to scale the magnitude of the
    update.
    """
    def __init__(self, embedding_model, device):
        """
        Initializes the GroundTruthSimulator.

        Args:
            embedding_model: The sentence transformer model to use for
                creating embeddings.
            device (torch.device): The device to use for tensor operations.
        """
        self.embedding_model = embedding_model
        self.device = device

    def get_target_delta_m(self, event_data, old_identity_embedding, new_identity_embedding):
        """
        Generates a target memory update (ΔM).

        The target ΔM is calculated as the sum of the change in the identity
        embedding and the event embedding, scaled by the significance of the
        event. The result is then projected to the same dimension as the memory
        space.

        Args:
            event_data (dict): The event data.
            old_identity_embedding (torch.Tensor): The user's identity
                embedding before the event.
            new_identity_embedding (torch.Tensor): The user's identity
                embedding after the event.

        Returns:
            torch.Tensor: The target memory update (ΔM).
        """
        significance = event_data.get("significance", 0)

        identity_change = new_identity_embedding - old_identity_embedding

        event_content = event_data.get("content", "")
        event_embedding = self.embedding_model.encode(event_content, convert_to_tensor=True).to(self.device)

        identity_dim = old_identity_embedding.shape[1]
        if event_embedding.shape[0] < identity_dim:
            padding = torch.zeros(identity_dim - event_embedding.shape[0], device=self.device)
            event_embedding = torch.cat((event_embedding, padding))
        elif event_embedding.shape[0] > identity_dim:
            event_embedding = event_embedding[:identity_dim]

        # Combine identity change and event context
        target_delta_m = (identity_change + event_embedding.unsqueeze(0)) * significance

        # Project the target_delta_m to the memory space dimension
        memory_dim = config.MEMORY_SIZE
        if target_delta_m.shape[1] < memory_dim:
            padding = torch.zeros(1, memory_dim - target_delta_m.shape[1], device=self.device)
            target_delta_m = torch.cat((target_delta_m, padding), dim=1)
        elif target_delta_m.shape[1] > memory_dim:
            target_delta_m = target_delta_m[:, :memory_dim]

        return target_delta_m
