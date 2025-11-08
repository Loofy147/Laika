import numpy as np
from . import config

class AdaptiveEventDetector:
    """
    Detects significant events in a stream of interactions.

    This class uses an adaptive threshold to determine whether an interaction
    is significant enough to be considered an event. The threshold is
    adjusted based on the running average of the significance of recent
    interactions.
    """
    def __init__(self):
        """
        Initializes the AdaptiveEventDetector.
        """
        self.threshold = config.EVENT_THRESHOLD
        self.window_size = config.WINDOW_SIZE
        self.adjustment_factor = config.ADJUSTMENT_FACTOR
        self.significance_history = []

    def _update_threshold(self):
        """
        Adjusts the threshold based on the running average of significance.

        If the significance history is larger than the window size, the
        threshold is updated to be the running average of the last
        `window_size` significance scores plus an adjustment factor.
        """
        if len(self.significance_history) > self.window_size:
            running_avg = np.mean(self.significance_history[-self.window_size:])
            self.threshold = running_avg + self.adjustment_factor

    def detect(self, interaction_data):
        """
        Detects if an interaction is significant enough to be an event.

        Args:
            interaction_data (dict): A dictionary containing information
                about the interaction, including its significance.

        Returns:
            dict: A dictionary representing the event if the interaction is
                  significant, otherwise None.
        """
        significance = interaction_data.get("significance", 0)
        self.significance_history.append(significance)
        self._update_threshold()

        event_type = "significant_interaction"
        if interaction_data.get('type') == 'identity_update':
            event_type = 'identity_update'

        if significance > self.threshold:
            return {"event_type": event_type, "data": interaction_data}
        return None
