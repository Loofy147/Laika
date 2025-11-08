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

        # Reset the agent manager before each test
        from ai_memory_system.api import agent_manager
        agent_manager.__init__()

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

    def test_model_weights_change_after_training(self):
        """Verify that the model's weights change after a training cycle."""
        from ai_memory_system.api import agent_manager

        headers = {'Authorization': 'Bearer test-token'}

        # 1. Trigger an interaction to create agent and log data
        interaction = {"type": "chat", "content": "This is a test interaction.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

        # 2. Get the agent instance and deepcopy the initial weights
        user_id = "user1"
        ai_agent = agent_manager.get_agent(user_id)
        initial_weights = copy.deepcopy(ai_agent.memory_controller.f_theta.state_dict())

        # 3. Trigger the training endpoint
        train_response = self.app.post('/train', headers=headers)
        self.assertEqual(train_response.status_code, 200)

        # 4. Get the new weights
        trained_weights = ai_agent.memory_controller.f_theta.state_dict()

        # 5. Compare the weights
        weights_have_changed = False
        for key in initial_weights:
            # Check if the tensors are not equal
            if not torch.equal(initial_weights[key], trained_weights[key]):
                weights_have_changed = True
                break

        self.assertTrue(weights_have_changed, "Model weights did not change after training.")

    def test_memory_update(self):
        """Verify that the memory state changes after an interaction."""
        from ai_memory_system.api import agent_manager

        headers = {'Authorization': 'Bearer test-token'}

        # Get the initial memory state
        response = self.app.get('/memory', headers=headers)
        self.assertEqual(response.status_code, 200)
        initial_memory = response.json['memory_state']

        # Trigger an interaction
        interaction = {"type": "chat", "content": "This is a test interaction.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

        # Get the new memory state
        response = self.app.get('/memory', headers=headers)
        self.assertEqual(response.status_code, 200)
        new_memory = response.json['memory_state']

        # Check that the memory state has changed
        self.assertNotEqual(initial_memory, new_memory)

    def test_identity_update(self):
        """Verify that the identity embedding changes after an identity update."""
        from ai_memory_system.api import agent_manager

        headers = {'Authorization': 'Bearer test-token'}

        # Trigger an interaction to create the agent
        interaction = {"type": "chat", "content": "This is a test interaction.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

        user_id = "user1"
        ai_agent = agent_manager.get_agent(user_id)
        initial_identity = ai_agent.identity.get_properties_tensor()

        # Update the identity
        new_properties = {"interests": ["python", "machine_learning"]}
        response = self.app.post('/identity', json=new_properties, headers=headers)
        self.assertEqual(response.status_code, 200)

        # Get the new identity
        new_identity = ai_agent.identity.get_properties_tensor()

        # Check that the identity embedding has changed
        self.assertFalse(torch.equal(initial_identity, new_identity))

    def test_interact_input_validation(self):
        """Verify that the /interact endpoint validates input."""
        headers = {'Authorization': 'Bearer test-token'}

        # Missing 'type'
        interaction = {"content": "Test input validation.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 400)

        # Missing 'content'
        interaction = {"type": "chat", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 400)

        # Missing 'significance'
        interaction = {"type": "chat", "content": "Test input validation."}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 400)

        # Valid request
        interaction = {"type": "chat", "content": "Test input validation.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_pydantic_validation(self):
        """Verify that the Pydantic models correctly validate input."""
        headers = {'Authorization': 'Bearer test-token'}

        # Invalid significance (greater than 1.0)
        interaction = {"type": "chat", "content": "Test pydantic validation.", "significance": 1.1}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 400)

        # Invalid significance (less than 0.0)
        interaction = {"type": "chat", "content": "Test pydantic validation.", "significance": -0.1}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 400)

        # Invalid type (not a string)
        interaction = {"type": 123, "content": "Test pydantic validation.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 400)

        # Valid request
        interaction = {"type": "chat", "content": "Test pydantic validation.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_replay_buffer_persistence(self):
        """Verify that the replay buffer is saved and loaded correctly."""
        from ai_memory_system.api import agent_manager

        headers = {'Authorization': 'Bearer test-token'}

        # 1. Trigger an interaction to create agent and populate replay buffer
        interaction = {"type": "chat", "content": "This should be in the replay buffer.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

        # 2. Get the agent and check the replay buffer
        user_id = "user1"
        ai_agent = agent_manager.get_agent(user_id)
        self.assertEqual(len(ai_agent.replay_buffer), 1)

        # 3. "Restart" the application by creating a new agent manager
        new_agent_manager = agent_manager.__class__()
        reloaded_ai_agent = new_agent_manager.get_agent(user_id)

        # 4. Check that the replay buffer was reloaded
        self.assertEqual(len(reloaded_ai_agent.replay_buffer), 1)
        self.assertEqual(reloaded_ai_agent.replay_buffer.sample(1)[0]['inputs']['event_tensor'][0][0],
                         ai_agent.replay_buffer.sample(1)[0]['inputs']['event_tensor'][0][0])

if __name__ == '__main__':
    unittest.main()
