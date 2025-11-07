import unittest
import torch
import json
import os
import copy
from ai_memory_system.core import MemoryAI
from ai_memory_system.api import app, DATA_DIR

class TestTrainingEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        # Clean up any state files and logs before running tests
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))

    def get_token(self, username="user1"):
        response = self.app.post('/login', json={"username": username})
        self.assertEqual(response.status_code, 200)
        return response.json['token']

    def test_training_pipeline(self):
        """Verify the full asynchronous training pipeline."""
        token = self.get_token()
        headers = {'Authorization': token}

        # 1. Make several calls to /interact
        interaction1 = {"type": "chat", "content": "First training interaction.", "significance": 0.9}
        interaction2 = {"type": "chat", "content": "Second training interaction.", "significance": 0.8}
        self.app.post('/interact', json=interaction1, headers=headers)
        self.app.post('/interact', json=interaction2, headers=headers)

        # 2. Check that a training log file is created
        user_id = "user1"
        log_path = os.path.join(DATA_DIR, f"{user_id}_training_log.jsonl")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

        # 3. Call /train and verify the model's loss changes
        # Get model state before training by creating a deep copy
        from ai_memory_system.api import agents
        agent_before = agents[user_id]
        state_before = copy.deepcopy(agent_before.memory_controller.f_theta.state_dict())

        train_response = self.app.post('/train', headers=headers)
        self.assertEqual(train_response.status_code, 200)
        self.assertIn("average_loss", train_response.json)

        # Get model state after training and check that weights have changed
        agent_after = agents[user_id]
        state_after = agent_after.memory_controller.f_theta.state_dict()

        weights_changed = False
        for key in state_before:
            if not torch.equal(state_before[key], state_after[key]):
                weights_changed = True
                break
        self.assertTrue(weights_changed, "Model weights did not change after training.")

        # 4. Confirm the training log is cleared after training
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, 'r') as f:
            self.assertEqual(len(f.readlines()), 0)

if __name__ == '__main__':
    unittest.main()
