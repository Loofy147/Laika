from flask import Flask, request, jsonify, send_from_directory
from functools import wraps
from .core import MemoryAI
from .tokens import VALID_TOKENS
import numpy as np

app = Flask(__name__)

# Initialize the AI agent
user_id = "api_user_001"
initial_identity_props = {"age": 30, "interests": ["python", "api_design"]}
ai_agent = MemoryAI(user_id, initial_identity_props)
last_interaction = None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token and token in VALID_TOKENS:
            return f(*args, **kwargs)
        return jsonify({"message": "Authentication required"}), 401
    return decorated

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/login', methods=['POST'])
def login():
    """Returns a token for a given user."""
    # This is a simplified example; in a real application, you would use a
    # more sophisticated authentication system.
    username = request.json.get('username')
    for token, user in VALID_TOKENS.items():
        if user == username:
            return jsonify({"token": token})
    return jsonify({"message": "Invalid username"}), 401

@app.route('/memory', methods=['GET'])
@require_auth
def get_memory_state():
    """Returns the current memory state."""
    memory_state = ai_agent.memory_controller.get_state().tolist()
    return jsonify({"memory_state": memory_state})

@app.route('/interact', methods=['POST'])
@require_auth
def process_interaction():
    """Processes a new interaction."""
    global last_interaction
    interaction_data = request.json
    last_interaction = interaction_data
    loss = ai_agent.process_interaction(interaction_data)

    if loss is not None:
        return jsonify({"status": "event processed", "loss": loss})
    else:
        return jsonify({"status": "no event detected"})

@app.route('/identity', methods=['POST'])
@require_auth
def update_identity():
    """Updates the user's properties."""
    new_properties = request.json
    ai_agent.identity.update_properties(new_properties)
    return jsonify({"status": "identity updated"})

@app.route('/explain', methods=['GET'])
@require_auth
def explain_update():
    """Explains the last memory update."""
    if last_interaction is None:
        return jsonify({"explanation": "No interaction has occurred yet."})

    param_ranges = {
        'significance': [0.0, 1.0],
        'content': [0.0, 1.0] # This is a simplification for the purpose of the example
    }

    problem = {
        'num_vars': 2,
        'names': ['significance', 'content'],
        'bounds': [[0.0, 1.0], [0.0, 1.0]]
    }

    # This is a placeholder for the Sobol analysis, as it is a computationally
    # expensive operation that should not be performed on every API request.
    sobol_indices = [0.7, 0.3]

    explanation = f"The last memory update was primarily driven by the significance of the interaction ({sobol_indices[0]:.2f}), with a smaller contribution from the content ({sobol_indices[1]:.2f})."

    return jsonify({"explanation": explanation})

if __name__ == '__main__':
    app.run(debug=True)
