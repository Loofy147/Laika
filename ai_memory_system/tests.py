import unittest
import torch
import json
import os
import shutil
import copy
from ai_memory_system.api import app, DATA_DIR, ARCHIVE_DIR, load_tokens_from_env

class TestCriticalGaps(unittest.TestCase):
    """
    Test suite for the AI Memory System API.

    This class contains tests for authentication, training log archiving,
    identity updates, model weight changes, memory updates, and input
    validation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a clean environment for the entire test class.

        This method is called once before any tests in the class are run. It
        sets the 'VALID_API_TOKENS' environment variable and configures the
        Flask app for testing.
        """
        os.environ['VALID_API_TOKENS'] = 'test-token:user1,another-token:user2'
        load_tokens_from_env()
        app.config['TESTING'] = True

    def setUp(self):
        """
        Set up a clean environment for each test.

        This method is called before each test is run. It cleans up and
        recreates the data and archive directories to ensure that each test
        starts with a clean state.
        """
        # Clean up and recreate directories
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(ARCHIVE_DIR)

        self.app = app.test_client()

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the environment after all tests.

        This method is called once after all tests in the class have been run.
        It cleans up the data directory and unsets the 'VALID_API_TOKENS'
        environment variable.
        """
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if 'VALID_API_TOKENS' in os.environ:
            del os.environ['VALID_API_TOKENS']

    def test_authentication_required(self):
        """
        Tests that the API endpoints are protected by authentication.
        """
        response = self.app.get('/memory')
        self.assertEqual(response.status_code, 401)

        response = self.app.get('/memory', headers={'Authorization': 'Bearer invalid-token'})
        self.assertEqual(response.status_code, 401)

        headers = {'Authorization': 'Bearer test-token'}
        response = self.app.get('/memory', headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_training_log_archiving(self):
        """
        Tests that the training log is archived after training.
        """
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
        """
        Tests that updating the user's identity affects the ground truth.
        """
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
        """
        Tests that the model's weights change after a training cycle.
        """
        from ai_memory_system.api import agents  # Import agents dict

        headers = {'Authorization': 'Bearer test-token'}

        # 1. Trigger an interaction to create agent and log data
        interaction = {"type": "chat", "content": "This is a test interaction.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

        # 2. Get the agent instance and deepcopy the initial weights
        user_id = "user1"
        ai_agent = agents[user_id]
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
        """
        Tests that the memory state is updated after an interaction.
        """
        from ai_memory_system.api import agents

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
        """
        Tests that the user's identity is updated correctly.
        """
        from ai_memory_system.api import agents

        headers = {'Authorization': 'Bearer test-token'}

        # Trigger an interaction to create the agent
        interaction = {"type": "chat", "content": "This is a test interaction.", "significance": 0.9}
        response = self.app.post('/interact', json=interaction, headers=headers)
        self.assertEqual(response.status_code, 200)

        user_id = "user1"
        ai_agent = agents[user_id]
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
        """
        Tests that the /interact endpoint validates its input.
        """
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

if __name__ == '__main__':
    unittest.main()
