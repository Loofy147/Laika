import numpy as np
from . import config

class AdaptiveEventDetector:
    """Detects events with an adaptive threshold."""
    def __init__(self):
        self.threshold = config.EVENT_THRESHOLD
        self.window_size = config.WINDOW_SIZE
        self.adjustment_factor = config.ADJUSTMENT_FACTOR
        self.significance_history = []

    def _update_threshold(self):
        """Adjusts the threshold based on the running average of significance."""
        if len(self.significance_history) > self.window_size:
            running_avg = np.mean(self.significance_history[-self.window_size:])
            self.threshold = running_avg + self.adjustment_factor

    def detect(self, interaction_data):
        """Detects if an interaction is significant enough to be an event."""
        significance = interaction_data.get("significance", 0)
        self.significance_history.append(significance)
        self._update_threshold()

        event_type = "significant_interaction"
        if interaction_data.get('type') == 'identity_update':
            event_type = 'identity_update'

        if significance > self.threshold:
            return {"event_type": event_type, "data": interaction_data}
        return None
