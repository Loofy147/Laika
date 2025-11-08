class EventDetector:
    """
    Simulates event detection based on interaction significance.

    This class provides a simple mechanism for detecting significant events
    based on a fixed threshold.
    """
    def __init__(self, event_threshold=0.5):
        """
        Initializes the EventDetector.

        Args:
            event_threshold (float, optional): The threshold for detecting
                significant events. Defaults to 0.5.
        """
        self.event_threshold = event_threshold

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
        if significance > self.event_threshold:
            return {"event_type": "significant_interaction", "data": interaction_data}
        return None
