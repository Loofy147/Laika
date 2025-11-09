import os
from .core import MemoryAI
from . import config

class AgentManager:
    """Manages the lifecycle of MemoryAI agent instances."""

    def __init__(self):
        self._agents = {}

    def get_agent(self, user_id: str) -> MemoryAI:
        """
        Retrieves an agent instance for a given user.
        Loads the agent from disk or creates a new one if not already in memory.
        """
        if user_id in self._agents:
            return self._agents[user_id]

        initial_identity_props = {"age": 30, "interests": ["python", "api_design"]}
        state_filepath = os.path.join(config.DATA_DIR, f"{user_id}.pt")
        training_log_path = os.path.join(config.DATA_DIR, f"{user_id}_training_log.jsonl")

        agent = MemoryAI(
            user_id,
            initial_identity_props,
            state_filepath=state_filepath,
            training_log_path=training_log_path
        )

        self._agents[user_id] = agent

        return agent
