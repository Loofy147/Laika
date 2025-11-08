from flask import Flask, request, jsonify, send_from_directory
from functools import wraps
from .core import MemoryAI
import numpy as np
import os
import torch
import json
from .lock_manager import UserStateLockManager
from .validation import InteractionRequest
from pydantic import ValidationError
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram

app = Flask(__name__)
metrics = PrometheusMetrics(app)
lock_manager = UserStateLockManager(timeout=5.0)

# Custom metrics
interaction_counter = Counter(
    'ai_interactions_total',
    'Total number of interactions',
    ['user_id', 'event_detected']
)

memory_norm_histogram = Histogram(
    'ai_memory_norm',
    'Memory state norm distribution'
)

training_loss_histogram = Histogram(
    'ai_training_loss',
    'Training loss distribution'
)

# Directory to store agent state
DATA_DIR = "agent_data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(ARCHIVE_DIR):
    os.makedirs(ARCHIVE_DIR)

# Load valid tokens from environment variable
VALID_TOKENS = {}
def load_tokens_from_env():
    global VALID_TOKENS
    VALID_TOKENS_str = os.environ.get('VALID_API_TOKENS', '')
    VALID_TOKENS = dict(token.split(':') for token in VALID_TOKENS_str.split(',') if ':' in token)

load_tokens_from_env()


# Dictionary to hold agent instances, keyed by user_id
agents = {}

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message": "Authorization header missing"}), 401

        try:
            token_type, token = auth_header.split()
            if token_type.lower() != 'bearer':
                return jsonify({"message": "Invalid token type"}), 401
        except ValueError:
            return jsonify({"message": "Invalid Authorization header format"}), 401

        if not token or token not in VALID_TOKENS:
            return jsonify({"message": "Authentication required"}), 401

        user_id = VALID_TOKENS[token]

        # Load or create agent for the user
        if user_id not in agents:
            initial_identity_props = {"age": 30, "interests": ["python", "api_design"]}
            state_filepath = os.path.join(DATA_DIR, f"{user_id}.pt")
            training_log_path = os.path.join(DATA_DIR, f"{user_id}_training_log.jsonl")
            agents[user_id] = MemoryAI(user_id, initial_identity_props, state_filepath=state_filepath, training_log_path=training_log_path)

        ai_agent = agents[user_id]

        return f(ai_agent, *args, **kwargs)
    return decorated

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/login', methods=['POST'])
def login():
    """Returns a token for a given user."""
    # This is a simplified example for demonstration.
    # In a real application, you would use a proper authentication provider.
    username = request.json.get('username')
    for token, user in VALID_TOKENS.items():
        if user == username:
            return jsonify({"token": token})
    return jsonify({"message": "Invalid username"}), 401

@app.route('/memory', methods=['GET'])
@require_auth
def get_memory_state(ai_agent):
    """Returns the current memory state."""
    memory_state = ai_agent.memory_controller.get_state().tolist()
    return jsonify({"memory_state": memory_state})

@app.route('/interact', methods=['POST'])
@require_auth
def process_interaction(ai_agent):
    """Processes a new interaction."""
    user_id = VALID_TOKENS[request.headers.get('Authorization').split()[1]]

    try:
        validated = InteractionRequest(**request.json)
        interaction_data = validated.dict()
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    ai_agent.last_interaction = interaction_data

    with lock_manager.user_lock(user_id):
        input_tensors = ai_agent.process_interaction(interaction_data)
        ai_agent.save_state()

    if input_tensors is not None:
        event_detected = True
        ai_agent.last_explanation_data = input_tensors
        status = "event processed, data logged for training"
    else:
        event_detected = False
        status = "no event detected"

    interaction_counter.labels(
        user_id=user_id,
        event_detected=str(event_detected)
    ).inc()

    memory_norm = torch.norm(ai_agent.memory_controller.get_state()).item()
    memory_norm_histogram.observe(memory_norm)

    return jsonify({"status": status})

@app.route('/identity', methods=['POST'])
@require_auth
def update_identity(ai_agent):
    """Updates the user's properties."""
    new_properties = request.json
    ai_agent.identity.update_properties(new_properties)
    return jsonify({"status": "identity updated"})

@app.route('/train', methods=['POST'])
@require_auth
def train_agent(ai_agent):
    """Triggers a training cycle on logged data."""
    if not ai_agent.training_log_path or not os.path.exists(ai_agent.training_log_path):
        return jsonify({"message": "No training data to process."}), 404

    with open(ai_agent.training_log_path, 'r') as f:
        batch_data = [json.loads(line) for line in f]

    if not batch_data:
        return jsonify({"message": "Training log is empty."}), 200

    with lock_manager.user_lock(ai_agent.identity.user_id):
        avg_loss = ai_agent.train_on_batch(batch_data)
        ai_agent.save_state()

    # Archive the log file
    archive_path = os.path.join(ARCHIVE_DIR, f"{ai_agent.identity.user_id}_{torch.randint(0, 100000, (1,)).item()}.jsonl")
    os.rename(ai_agent.training_log_path, archive_path)

    return jsonify({"status": "training complete", "average_loss": avg_loss})

@app.route('/explain', methods=['GET'])
@require_auth
def explain_update(ai_agent):
    """Explains the last memory update using gradient-based feature importance."""
    if ai_agent.last_explanation_data is None:
        return jsonify({"explanation": "No memory update has occurred yet for which an explanation can be generated."})

    input_tensors = ai_agent.last_explanation_data

    for key, tensor in input_tensors.items():
        input_tensors[key] = tensor.clone().detach().requires_grad_(True)

    predicted_delta_m = ai_agent.memory_controller.predict_delta_m(**input_tensors)

    scalar_output = torch.norm(predicted_delta_m)
    scalar_output.backward()

    grads = {
        "memory": torch.norm(input_tensors["memory_state"].grad).item(),
        "identity": torch.norm(input_tensors["identity_tensor"].grad).item(),
        "event": torch.norm(input_tensors["event_tensor"].grad).item()
    }

    total_grad = sum(grads.values()) if sum(grads.values()) > 0 else 1
    importances = {name: (grad / total_grad) * 100 for name, grad in grads.items()}

    explanation = (
        f"The last memory update was influenced by the following factors:\n"
        f"- Current Memory State: {importances['memory']:.2f}%\n"
        f"- User Identity: {importances['identity']:.2f}%\n"
        f"- Interaction Event: {importances['event']:.2f}%"
    )

    return jsonify({"explanation": explanation})

if __name__ == '__main__':
    app.run(debug=True)
