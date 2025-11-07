import unittest
import torch
import json
from ai_memory_system.core import MemoryAI
from ai_memory_system.api import app

class TestMemoryAI(unittest.TestCase):
    def setUp(self):
        self.user_id = "test_user"
        self.initial_identity_props = {"age": 25, "interests": ["coding", "reading"]}
        self.ai = MemoryAI(self.user_id, self.initial_identity_props)
        self.app = app.test_client()
        self.token = self.get_token()

    def get_token(self):
        response = self.app.post('/login', json={"username": "user1"})
        return response.json['token']

    def test_smoke(self):
        interaction = {"type": "chat", "content": "Hello, world!", "significance": 0.9}
        loss = self.ai.process_interaction(interaction)
        self.assertIsNotNone(loss)

    def test_api_memory(self):
        response = self.app.get('/memory', headers={'Authorization': self.token})
        self.assertEqual(response.status_code, 200)
        self.assertIn('memory_state', response.json)

    def test_api_interact(self):
        interaction = {"type": "chat", "content": "Hello, world!", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers={'Authorization': self.token})
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', response.json)

    def test_api_explain(self):
        interaction = {"type": "chat", "content": "Hello, world!", "significance": 0.9}
        self.app.post('/interact', json=interaction, headers={'Authorization': self.token})
        response = self.app.get('/explain', headers={'Authorization': self.token})
        self.assertEqual(response.status_code, 200)
        self.assertIn('explanation', response.json)

    def test_api_identity(self):
        new_properties = {"interests": ["coding", "reading", "music"]}
        response = self.app.post('/identity', json=new_properties, headers={'Authorization': self.token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'identity updated')

if __name__ == '__main__':
    unittest.main()
