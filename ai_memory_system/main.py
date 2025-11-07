import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from .core import MemoryAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_memory_norm(memory_norms):
    """Plots the memory norm over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(memory_norms, label='Memory Norm')
    plt.xlabel('Interaction Step')
    plt.ylabel('Norm')
    plt.title('Memory Norm Over Time')
    plt.legend()
    plt.savefig('memory_norm.png')

def run_simulation(ai_agent, interactions):
    """Runs the simulation loop."""
    logging.info("Starting simulation...")
    memory_norms = []
    for i, interaction in enumerate(interactions):
        logging.info(f"--- Processing Interaction {i+1}: '{interaction['content']}' ---")

        old_norm = torch.norm(ai_agent.memory_controller.get_state()).item()
        loss = ai_agent.process_interaction(interaction)
        new_norm = torch.norm(ai_agent.memory_controller.get_state()).item()
        memory_norms.append(new_norm)

        if loss is not None:
            logging.info(f"  Loss after training: {loss:.6f}")
        logging.info(f"  Memory state norm changed from {old_norm:.4f} to {new_norm:.4f}")

    logging.info("Simulation finished.")
    logging.info(f"Final Memory State Norm: {torch.norm(ai_agent.memory_controller.get_state()).item():.4f}")
    plot_memory_norm(memory_norms)

if __name__ == "__main__":
    user_id = "user_xyz_123"
    initial_identity_props = {"age": 35, "interests": ["artificial intelligence", "philosophy"]}

    try:
        ai_agent = MemoryAI(user_id, initial_identity_props)

        interactions = [
            {"type": "chat", "content": "What is the meaning of life?", "significance": 0.8},
            {"type": "chat", "content": "Tell me a joke.", "significance": 0.4},
            {"type": "feedback", "content": "Your previous answer was insightful.", "significance": 0.9},
            {"type": "update", "content": "I recently read a book on quantum physics.", "significance": 0.7},
        ]

        run_simulation(ai_agent, interactions)

        # Run sensitivity analysis
        param_ranges = {
            'lambda_decay': [0.0, 0.1],
            'activation_factor': [0.5, 1.5]
        }
        #Si = ai_agent.analyze_sensitivity(param_ranges)
        #logging.info(f"Sobol Indices: {Si}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
