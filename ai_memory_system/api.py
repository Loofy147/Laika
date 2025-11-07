from flask import Flask, request, jsonify
from .core import MemoryAI

app = Flask(__name__)

# Initialize the AI agent
user_id = "api_user_001"
initial_identity_props = {"age": 30, "interests": ["python", "api_design"]}
ai_agent = MemoryAI(user_id, initial_identity_props)

@app.route('/memory', methods=['GET'])
def get_memory_state():
    """Returns the current memory state."""
    memory_state = ai_agent.memory_controller.get_state().tolist()
    return jsonify({"memory_state": memory_state})

@app.route('/interact', methods=['POST'])
def process_interaction():
    """Processes a new interaction."""
    interaction_data = request.json
    loss = ai_agent.process_interaction(interaction_data)

    if loss is not None:
        return jsonify({"status": "event processed", "loss": loss})
    else:
        return jsonify({"status": "no event detected"})

if __name__ == '__main__':
    app.run(debug=True)
