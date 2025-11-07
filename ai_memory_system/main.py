import torch
from .core import MemoryAI

if __name__ == "__main__":
    user_id = "user_abc_789"
    initial_identity_props = {"age": 28, "interests": ["machine_learning", "history"]}
    ai_agent = MemoryAI(user_id, initial_identity_props)

    interactions = [
        {"type": "chat", "content": "Just passing by", "significance": 0.2},
        {"type": "chat", "content": "Tell me more about continuous learning.", "significance": 0.7},
        {"type": "feedback", "content": "That last explanation was very helpful!", "significance": 0.9},
        {"type": "chat", "content": "What's the weather like?", "significance": 0.3},
        {"type": "update", "content": "I've started a new hobby: painting.", "significance": 0.8},
    ]

    for i, interaction in enumerate(interactions):
        print(f"\n--- Processing Interaction {i+1} ---")
        old_norm = torch.norm(ai_agent.memory.get_state()).item()
        loss = ai_agent.process_interaction(interaction)
        new_norm = torch.norm(ai_agent.memory.get_state()).item()
        if loss is not None:
            print(f"  Loss after training: {loss:.6f}")
        print(f"  Memory state norm changed from {old_norm:.4f} to {new_norm:.4f}")

    print("\nFinal Memory State Norm:", torch.norm(ai_agent.memory.get_state()).item())
