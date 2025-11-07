class EventDetector:
    """Simulates event detection based on interaction significance."""
    def __init__(self, event_threshold=0.5):
        self.event_threshold = event_threshold

    def detect(self, interaction_data):
        """Detects if an interaction is significant enough to be an event."""
        significance = interaction_data.get("significance", 0)
        if significance > self.event_threshold:
            return {"event_type": "significant_interaction", "data": interaction_data}
        return None
