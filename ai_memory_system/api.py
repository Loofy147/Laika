from flask import Flask, request, jsonify, send_from_directory
from functools import wraps
from .core import MemoryAI
from .tokens import VALID_TOKENS
import numpy as np
import os
import torch

app = Flask(__name__)

# Directory to store agent state
DATA_DIR = "agent_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Dictionary to hold agent instances, keyed by user_id
agents = {}
last_interactions = {}
last_explanation_data = {}

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or token not in VALID_TOKENS:
            return jsonify({"message": "Authentication required"}), 401

        user_id = VALID_TOKENS[token]

        # Load or create agent for the user
        if user_id not in agents:
            # A real app would load user properties from a database
            initial_identity_props = {"age": 30, "interests": ["python", "api_design"]}
            state_filepath = os.path.join(DATA_DIR, f"{user_id}.pt")
            agents[user_id] = MemoryAI(user_id, initial_identity_props, state_filepath=state_filepath)

        ai_agent = agents[user_id]

        return f(ai_agent, *args, **kwargs)
    return decorated

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/login', methods=['POST'])
def login():
    """Returns a token for a given user."""
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
    user_id = VALID_TOKENS[request.headers.get('Authorization')]
    interaction_data = request.json
    last_interactions[user_id] = interaction_data

    result = ai_agent.process_interaction(interaction_data)

    if result is not None:
        loss, input_tensors = result
        last_explanation_data[user_id] = input_tensors
        # Save agent state after interaction
        ai_agent.save_state()
        return jsonify({"status": "event processed", "loss": loss})
    else:
        # Save agent state even if no event was detected (e.g. adaptive threshold update)
        ai_agent.save_state()
        return jsonify({"status": "no event detected"})


@app.route('/identity', methods=['POST'])
@require_auth
def update_identity(ai_agent):
    """Updates the user's properties."""
    new_properties = request.json
    ai_agent.identity.update_properties(new_properties)
    # Also save state after identity update
    ai_agent.save_state()
    return jsonify({"status": "identity updated"})

@app.route('/explain', methods=['GET'])
@require_auth
def explain_update(ai_agent):
    """Explains the last memory update using gradient-based feature importance."""
    user_id = VALID_TOKENS[request.headers.get('Authorization')]

    if user_id not in last_explanation_data:
        return jsonify({"explanation": "No memory update has occurred yet for which an explanation can be generated."})

    input_tensors = last_explanation_data[user_id]

    # Ensure tensors require gradients
    for key, tensor in input_tensors.items():
        input_tensors[key] = tensor.clone().detach().requires_grad_(True)

    # Re-run the forward pass to build the computation graph
    predicted_delta_m = ai_agent.memory_controller.predict_delta_m(**input_tensors)

    # We need a scalar value to backpropagate from, so we use the norm of the output.
    scalar_output = torch.norm(predicted_delta_m)
    scalar_output.backward()

    # Get the gradients
    grads = {
        "memory": torch.norm(input_tensors["memory_state"].grad).item(),
        "identity": torch.norm(input_tensors["identity_tensor"].grad).item(),
        "event": torch.norm(input_tensors["event_tensor"].grad).item()
    }

    # Normalize the gradients to get feature importances
    total_grad = sum(grads.values())
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
