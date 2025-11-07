import numpy as np

class AdaptiveEventDetector:
    """Detects events with an adaptive threshold."""
    def __init__(self, initial_threshold=0.5, window_size=10, adjustment_factor=0.1):
        self.threshold = initial_threshold
        self.window_size = window_size
        self.adjustment_factor = adjustment_factor
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

        if significance > self.threshold:
            return {"event_type": "significant_interaction", "data": interaction_data}
        return None
