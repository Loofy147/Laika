import unittest
from .core import MemoryAI

class TestMemoryAI(unittest.TestCase):
    def test_initialization(self):
        """Tests that the MemoryAI class can be initialized without errors."""
        user_id = "test_user"
        initial_identity_props = {"age": 25, "interests": ["testing"]}
        try:
            ai_agent = MemoryAI(user_id, initial_identity_props)
            self.assertIsNotNone(ai_agent, "AI agent should not be None")
        except Exception as e:
            self.fail(f"MemoryAI initialization failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
