import unittest
import torch
from ai_memory_system.core import MemoryAI

class TestMemoryAI(unittest.TestCase):
    def test_smoke(self):
        user_id = "test_user"
        initial_identity_props = {"age": 25, "interests": ["coding", "reading"]}
        ai = MemoryAI(user_id, initial_identity_props)
        interaction = {"type": "chat", "content": "Hello, world!", "significance": 0.9}
        loss = ai.process_interaction(interaction)
        self.assertIsNotNone(loss)

if __name__ == '__main__':
    unittest.main()
