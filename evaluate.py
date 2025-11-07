import torch
import numpy as np
import logging
from ai_memory_system.core import MemoryAI

logging.basicConfig(level=logging.INFO)

def evaluate_memory_fidelity(ai_agent, interactions):
    """Evaluates the memory fidelity of the AI agent."""
    target_delta_m_norms = []
    predicted_delta_m_norms = []

    for interaction in interactions:
        event = ai_agent.event_detector.detect(interaction)
        if event:
            old_identity_embedding = ai_agent.identity.get_properties_tensor()
            if interaction.get('type') == 'identity_update':
                ai_agent.identity.update_properties(interaction)
            new_identity_embedding = ai_agent.identity.get_properties_tensor()

            memory_state = ai_agent.memory_controller.get_state()
            input_tensors = ai_agent._prepare_input_tensors(memory_state, old_identity_embedding, event['data'])

            target_delta_m = ai_agent.ground_truth_simulator.get_target_delta_m(event['data'], old_identity_embedding, new_identity_embedding)
            predicted_delta_m = ai_agent.memory_controller.predict_delta_m(**input_tensors)

            target_delta_m_norms.append(torch.norm(target_delta_m).item())
            predicted_delta_m_norms.append(torch.norm(predicted_delta_m).item())

    if not predicted_delta_m_norms:
        return 0.0
    return np.mean(np.abs(np.array(target_delta_m_norms) - np.array(predicted_delta_m_norms)))

def evaluate_learning_stability(ai_agent, interactions):
    """Evaluates the learning stability of the AI agent."""
    losses = []
    for interaction in interactions:
        result = ai_agent.process_interaction(interaction)
        if result:
            # This is a simplification; in a real scenario, you would train and get a loss
            pass

    # This metric is no longer meaningful in an async training world without modification
    # Returning a placeholder value
    return 0.0

def main():
    """Main function."""
    user_id = "eval_user"
    initial_identity_props = {"age": 30, "interests": ["ai", "evaluation"]}
    ai = MemoryAI(user_id, initial_identity_props)

    interactions = [
        {"type": "chat", "content": "Tell me about memory fidelity.", "significance": 0.8},
        {"type": "identity_update", "content": "My new interest is philosophy", "significance": 0.9, "interests": ["ai", "philosophy"]},
        {"type": "chat", "content": "What are the key metrics?", "significance": 0.7},
    ]

    memory_fidelity = evaluate_memory_fidelity(ai, interactions)
    learning_stability = evaluate_learning_stability(ai, interactions)

    logging.info(f"Memory Fidelity: {memory_fidelity:.4f}")
    logging.info(f"Learning Stability: {learning_stability:.4f} (Note: Placeholder value)")

if __name__ == "__main__":
    main()
