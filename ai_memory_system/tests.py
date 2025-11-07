import unittest
import torch
import json
import os
import shutil
import copy
from ai_memory_system.api import app, DATA_DIR, ARCHIVE_DIR, load_tokens_from_env

class TestCriticalGaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a clean environment for the entire test class."""
        os.environ['VALID_API_TOKENS'] = 'test-token:user1,another-token:user2'
        load_tokens_from_env()
        app.config['TESTING'] = True

    def setUp(self):
        """Set up a clean environment for each test."""
        # Clean up and recreate directories
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(ARCHIVE_DIR)

        self.app = app.test_client()

    @classmethod
    def tearDownClass(cls):
        """Clean up environment after all tests."""
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if 'VALID_API_TOKENS' in os.environ:
            del os.environ['VALID_API_TOKENS']

    def test_authentication_required(self):
        """Verify that endpoints are protected and require a valid token."""
        response = self.app.get('/memory')
        self.assertEqual(response.status_code, 401)

        response = self.app.get('/memory', headers={'Authorization': 'Bearer invalid-token'})
        self.assertEqual(response.status_code, 401)

        headers = {'Authorization': 'Bearer test-token'}
        response = self.app.get('/memory', headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_training_log_archiving(self):
        """Verify that the training log is archived, not deleted."""
        headers = {'Authorization': 'Bearer test-token'}

        interaction = {"type": "chat", "content": "Test log archiving.", "significance": 0.9}
        self.app.post('/interact', json=interaction, headers=headers)

        user_id = "user1"
        log_path = os.path.join(DATA_DIR, f"{user_id}_training_log.jsonl")
        self.assertTrue(os.path.exists(log_path), "Log file was not created.")

        train_response = self.app.post('/train', headers=headers)
        self.assertEqual(train_response.status_code, 200)

        self.assertFalse(os.path.exists(log_path), "Log file was not removed after training.")

        archive_files = os.listdir(ARCHIVE_DIR)
        self.assertEqual(len(archive_files), 1, "Log file was not archived.")
        self.assertTrue(archive_files[0].startswith(user_id))

    def test_identity_update_affects_ground_truth(self):
        """Verify that an identity update changes the target delta_m."""
        headers = {'Authorization': 'Bearer test-token'}

        interaction1 = {"type": "chat", "content": "A normal message.", "significance": 0.8}
        self.app.post('/interact', json=interaction1, headers=headers)

        interaction2 = {
            "type": "identity_update",
            "content": "I have a new interest: philosophy.",
            "significance": 0.9,
            "interests": ["coding", "philosophy"]
        }
        self.app.post('/interact', json=interaction2, headers=headers)

        log_path = os.path.join(DATA_DIR, "user1_training_log.jsonl")
        with open(log_path, 'r') as f:
            lines = [json.loads(line) for line in f.readlines()]

        self.assertEqual(len(lines), 2)

        target1_norm = torch.norm(torch.tensor(lines[0]['target']))
        target2_norm = torch.norm(torch.tensor(lines[1]['target']))

        self.assertNotAlmostEqual(target1_norm.item(), target2_norm.item(), delta=1e-5, msg="Identity update did not significantly change the ground truth target.")

if __name__ == '__main__':
    unittest.main()
