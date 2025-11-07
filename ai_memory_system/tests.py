import unittest
import torch
import json
import os
from ai_memory_system.core import MemoryAI
from ai_memory_system.api import app, DATA_DIR

class TestMemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        # Clean up any state files before running tests
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))

    def get_token(self, username="user1"):
        response = self.app.post('/login', json={"username": username})
        self.assertEqual(response.status_code, 200)
        return response.json['token']

    def test_memory_is_persisted(self):
        """Verify that memory state is saved and loaded across requests."""
        token = self.get_token()
        headers = {'Authorization': token}

        # First interaction
        interaction1 = {"type": "chat", "content": "This is the first interaction.", "significance": 0.9}
        response1 = self.app.post('/interact', json=interaction1, headers=headers)
        self.assertEqual(response1.status_code, 200)

        # Get the memory state
        response_mem1 = self.app.get('/memory', headers=headers)
        self.assertEqual(response_mem1.status_code, 200)
        memory_state1 = response_mem1.json['memory_state']

        # Simulate a server restart by creating a new test client
        # In a real scenario, the `agents` dictionary would be cleared.
        # Here, we can simulate it by clearing the agents dict in the app context.
        from ai_memory_system.api import agents
        agents.clear()

        # Second interaction with a new client instance for the same user
        app2 = app.test_client()
        response2 = app2.get('/memory', headers=headers)
        self.assertEqual(response2.status_code, 200)
        memory_state2 = response2.json['memory_state']

        # The memory states should be identical because the state was loaded from the file
        self.assertEqual(memory_state1, memory_state2)

    def test_different_users_have_different_memories(self):
        """Verify that two different users have independent memory states."""
        # User 1 interaction
        token1 = self.get_token(username="user1")
        headers1 = {'Authorization': token1}
        interaction1 = {"type": "chat", "content": "Hello from user 1", "significance": 0.9}
        self.app.post('/interact', json=interaction1, headers=headers1)

        response_mem1 = self.app.get('/memory', headers=headers1)
        memory_state1 = response_mem1.json['memory_state']

        # User 2 interaction
        token2 = self.get_token(username="user2")
        headers2 = {'Authorization': token2}
        interaction2 = {"type": "chat", "content": "Hello from user 2", "significance": 0.8}
        self.app.post('/interact', json=interaction2, headers=headers2)

        response_mem2 = self.app.get('/memory', headers=headers2)
        memory_state2 = response_mem2.json['memory_state']

        # The memory states should be different
        self.assertNotEqual(memory_state1, memory_state2)

        # Verify user 1's memory hasn't changed
        response_mem1_after = self.app.get('/memory', headers=headers1)
        memory_state1_after = response_mem1_after.json['memory_state']
        self.assertEqual(memory_state1, memory_state1_after)

if __name__ == '__main__':
    unittest.main()
