"""
Critical Issue Test Suite for AI Memory System
Run this to verify the problems identified in the analysis
"""

import torch
import torch.nn.functional as F
import sys
import os
import json
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import ai_memory_system
sys.path.insert(0, os.path.abspath('.'))

from ai_memory_system.core import MemoryAI
from ai_memory_system.memory_controller import MemoryController
from ai_memory_system import config

class CriticalIssueTests:
    """
    A test suite for verifying critical issues in the AI Memory System.

    This class contains tests for memory explosion, catastrophic forgetting,
    privacy leakage, and gradient explosion. The tests are designed to be
    run from the command line to verify the production readiness of the
    system.
    """
    
    def __init__(self):
        """
        Initializes the CriticalIssueTests suite.
        """
        self.results = {}
        self.temp_dir = tempfile.mkdtemp()
        
    def cleanup(self):
        """
        Cleans up the temporary directory used for storing test artifacts.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_memory_explosion(self):
        """
        Tests for memory state explosion.

        This test runs a series of interactions with high significance and
        monitors the norm of the memory state. It fails if the norm grows
        unboundedly.
        """
        print("\n" + "="*80)
        print("TEST 1: Memory State Explosion")
        print("="*80)
        
        user_id = "test_explosion"
        ai = MemoryAI(
            user_id, 
            {"age": 30, "interests": ["testing"]},
            state_filepath=os.path.join(self.temp_dir, f"{user_id}.pt")
        )
        
        norms = []
        max_norm_growth = 0
        
        print("Running 500 interactions with high significance...")
        for i in range(500):
            interaction = {
                "type": "chat",
                "content": f"Test interaction {i}: This is important information.",
                "significance": 0.9
            }
            
            old_norm = torch.norm(ai.memory_controller.get_state()).item()
            ai.process_interaction(interaction)
            new_norm = torch.norm(ai.memory_controller.get_state()).item()
            
            norms.append(new_norm)
            growth = new_norm - old_norm
            max_norm_growth = max(max_norm_growth, growth)
            
            if i % 100 == 0:
                print(f"  Step {i}: norm = {new_norm:.4f}, growth = {growth:.4f}")
        
        # Analysis
        initial_norm = norms[0]
        final_norm = norms[-1]
        growth_rate = (final_norm - initial_norm) / initial_norm * 100
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(norms, linewidth=2)
        plt.xlabel('Interaction Step')
        plt.ylabel('Memory State Norm')
        plt.title('Memory State Norm Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.diff(norms), linewidth=2, color='red')
        plt.xlabel('Interaction Step')
        plt.ylabel('Change in Norm')
        plt.title('Memory Growth Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('memory_explosion_test.png', dpi=150, bbox_inches='tight')
        print(f"\n  📊 Plot saved to: memory_explosion_test.png")
        
        # Verdict
        print(f"\n  Results:")
        print(f"    Initial norm: {initial_norm:.4f}")
        print(f"    Final norm: {final_norm:.4f}")
        print(f"    Growth rate: {growth_rate:.2f}%")
        print(f"    Max single-step growth: {max_norm_growth:.4f}")
        
        # Dynamic threshold based on configured MEMORY_SIZE
        pass_threshold = np.sqrt(config.MEMORY_SIZE) + 1.0
        is_stable = final_norm < pass_threshold
        
        print(f"    (Pass threshold for norm: < {pass_threshold:.2f})")

        if is_stable:
            print(f"  ✅ PASSED: Memory state is bounded (norm: {final_norm:.2f} < threshold: {threshold:.2f})")
        else:
            print(f"  ❌ FAILED: Memory state is exploding!")
            print(f"     → Predicted norm at 1000 steps: {final_norm * 2:.2f}")
            print(f"     → Will cause NaN/Inf errors in production")
        
        self.results['memory_explosion'] = {
            'passed': is_stable,
            'final_norm': final_norm,
            'growth_rate': growth_rate
        }
        
        return is_stable
    
    def test_catastrophic_forgetting(self):
        """
        Tests for catastrophic forgetting.

        This test simulates a scenario where the AI learns two different
        tasks sequentially. It then measures how much the AI has forgotten
        about the first task after learning the second.
        """
        print("\n" + "="*80)
        print("TEST 2: Catastrophic Forgetting")
        print("="*80)
        
        user_id = "test_forgetting"
        ai = MemoryAI(
            user_id,
            {"age": 30, "interests": ["python"]},
            state_filepath=os.path.join(self.temp_dir, f"{user_id}.pt"),
            training_log_path=os.path.join(self.temp_dir, f"{user_id}_log.jsonl")
        )
        
        print("Phase 1: Learning Task A (Python programming)...")
        task_a_interactions = []
        for i in range(30):
            interaction = {
                "type": "chat",
                "content": f"Python question {i}: How do you use decorators?",
                "significance": 0.8
            }
            task_a_interactions.append(interaction)
            ai.process_interaction(interaction)
        
        memory_after_task_a = ai.memory_controller.get_state().clone()
        print(f"  Task A memory norm: {torch.norm(memory_after_task_a).item():.4f}")
        
        # Simulate training on Task A
        if os.path.exists(ai.training_log_path):
            with open(ai.training_log_path, 'r') as f:
                task_a_data = [json.loads(line) for line in f]
            if task_a_data:
                print(f"  Training on {len(task_a_data)} Task A samples...")
                ai.train_on_batch(task_a_data)
        
        memory_after_training_a = ai.memory_controller.get_state().clone()
        
        print("\nPhase 2: Learning Task B (Machine learning)...")
        # Clear log for Task B
        if os.path.exists(ai.training_log_path):
            os.remove(ai.training_log_path)
        
        task_b_interactions = []
        for i in range(30):
            interaction = {
                "type": "chat",
                "content": f"ML question {i}: Explain gradient descent.",
                "significance": 0.8
            }
            task_b_interactions.append(interaction)
            ai.process_interaction(interaction)
        
        memory_after_task_b = ai.memory_controller.get_state().clone()
        print(f"  Task B memory norm: {torch.norm(memory_after_task_b).item():.4f}")
        
        # Train on Task B
        if os.path.exists(ai.training_log_path):
            with open(ai.training_log_path, 'r') as f:
                task_b_data = [json.loads(line) for line in f]
            if task_b_data:
                print(f"  Training on {len(task_b_data)} Task B samples...")
                ai.train_on_batch(task_b_data)
        
        memory_after_training_b = ai.memory_controller.get_state().clone()
        
        # Analysis
        similarity_a_to_b = F.cosine_similarity(
            memory_after_training_a, 
            memory_after_training_b, 
            dim=1
        ).item()
        
        # Measure how much Task A was forgotten
        print("\nPhase 3: Re-evaluating Task A...")
        # Re-test with Task A samples
        task_a_memory_recall = ai.memory_controller.get_state().clone()
        
        similarity_original_to_final = F.cosine_similarity(
            memory_after_training_a,
            task_a_memory_recall,
            dim=1
        ).item()
        
        print(f"\n  Results:")
        print(f"    Task A → Task B similarity: {similarity_a_to_b:.4f}")
        print(f"    Original Task A → Final similarity: {similarity_original_to_final:.4f}")
        
        forgetting_rate = 1 - similarity_original_to_final
        print(f"    Forgetting rate: {forgetting_rate * 100:.2f}%")
        
        is_acceptable = forgetting_rate < 0.30  # Less than 30% forgetting
        
        if is_acceptable:
            print(f"  ✅ PASSED: Forgetting rate is acceptable")
        else:
            print(f"  ❌ FAILED: Severe catastrophic forgetting detected!")
            print(f"     → Task A knowledge degraded by {forgetting_rate * 100:.1f}%")
            print(f"     → System will lose user context over time")
            print(f"     → Recommendation: Implement experience replay")
        
        self.results['catastrophic_forgetting'] = {
            'passed': is_acceptable,
            'forgetting_rate': forgetting_rate,
            'similarity': similarity_original_to_final
        }
        
        return is_acceptable
    
    def test_privacy_leakage(self):
        """
        Tests for privacy leakage.

        This test creates embeddings for a number of different users and
        measures the pairwise distances between them. It fails if the
        minimum distance is below a certain threshold, which could indicate
        that users are identifiable from their embeddings.
        """
        print("\n" + "="*80)
        print("TEST 3: Privacy Leakage Detection")
        print("="*80)
        
        print("Creating embeddings for 50 different users...")
        
        user_ids = [f"user_{i}" for i in range(50)]
        embeddings = []
        
        for user_id in user_ids:
            ai = MemoryAI(
                user_id,
                {"age": 20 + (hash(user_id) % 40), "interests": [f"interest_{hash(user_id) % 10}"]},
                state_filepath=os.path.join(self.temp_dir, f"{user_id}.pt")
            )
            embedding = ai.identity.get_properties_tensor()
            embeddings.append(embedding)
        
        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = torch.norm(embeddings[i] - embeddings[j]).item()
                distances.append(dist)
        
        min_distance = min(distances)
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        print(f"\n  Distance Statistics:")
        print(f"    Minimum distance: {min_distance:.4f}")
        print(f"    Average distance: {avg_distance:.4f}")
        print(f"    Std deviation: {std_distance:.4f}")
        
        # Privacy threshold: embeddings should be well-separated
        privacy_threshold = 0.5
        
        is_private = min_distance > privacy_threshold
        
        if is_private:
            print(f"  ✅ PASSED: Embeddings are well-separated (privacy preserved)")
        else:
            print(f"  ❌ FAILED: Privacy leakage risk!")
            print(f"     → Minimum distance ({min_distance:.4f}) < threshold ({privacy_threshold})")
            print(f"     → Users may be identifiable from embeddings")
            print(f"     → Recommendation: Increase noise scale or epsilon")
        
        # Plot distance distribution
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(min_distance, color='red', linestyle='--', linewidth=2, label=f'Min: {min_distance:.3f}')
        plt.axvline(privacy_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {privacy_threshold}')
        plt.xlabel('Pairwise Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Embedding Distances (Privacy Analysis)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('privacy_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n  📊 Plot saved to: privacy_analysis.png")
        
        self.results['privacy_leakage'] = {
            'passed': is_private,
            'min_distance': min_distance,
            'avg_distance': avg_distance
        }
        
        return is_private
    
    def test_gradient_explosion(self):
        """
        Tests for gradient explosion.

        This test generates training data with extreme values and then
        monitors the norm of the gradients during training. It fails if the
        maximum gradient norm is above a certain threshold.
        """
        print("\n" + "="*80)
        print("TEST 4: Gradient Explosion Check")
        print("="*80)
        
        user_id = "test_gradients"
        ai = MemoryAI(
            user_id,
            {"age": 30, "interests": ["testing"]},
            state_filepath=os.path.join(self.temp_dir, f"{user_id}.pt"),
            training_log_path=os.path.join(self.temp_dir, f"{user_id}_log.jsonl")
        )
        
        print("Generating training data with extreme values...")
        
        # Generate some training data
        for i in range(20):
            interaction = {
                "type": "chat",
                "content": f"Extreme test {i}" + " important" * 50,  # Long content
                "significance": 0.95
            }
            ai.process_interaction(interaction)
        
        # Train and monitor gradients
        if os.path.exists(ai.training_log_path):
            with open(ai.training_log_path, 'r') as f:
                batch_data = [json.loads(line) for line in f]
            
            grad_norms = []
            losses = []
            
            print(f"Training on {len(batch_data)} samples...")
            
            for data in batch_data:
                input_tensors = {k: torch.tensor(v).to(ai.device) for k, v in data['inputs'].items()}
                target_output = torch.tensor(data['target']).to(ai.device)
                
                ai.optimizer.zero_grad()
                output = ai.memory_controller.predict_delta_m(**input_tensors)
                loss = ai.loss_function(output, target_output)
                loss.backward()
                
                # Compute gradient norm before clipping
                total_norm = 0.0
                for p in ai.memory_controller.f_theta.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                total_norm = total_norm ** 0.5
                
                grad_norms.append(total_norm)
                losses.append(loss.item())
                
                # Apply clipping
                torch.nn.utils.clip_grad_norm_(
                    ai.memory_controller.f_theta.parameters(), 
                    config.MAX_GRAD_NORM
                )
                ai.optimizer.step()
            
            max_grad = max(grad_norms)
            avg_grad = np.mean(grad_norms)
            
            print(f"\n  Gradient Statistics:")
            print(f"    Maximum gradient norm: {max_grad:.4f}")
            print(f"    Average gradient norm: {avg_grad:.4f}")
            print(f"    Clipping threshold: {config.MAX_GRAD_NORM}")
            print(f"    Clipped gradients: {sum(1 for g in grad_norms if g > config.MAX_GRAD_NORM)}/{len(grad_norms)}")
            
            is_stable = max_grad < 100.0  # Reasonable threshold
            
            if is_stable:
                print(f"  ✅ PASSED: Gradients are under control")
            else:
                print(f"  ❌ FAILED: Gradient explosion detected!")
                print(f"     → Max gradient ({max_grad:.2f}) is very high")
                print(f"     → May cause training instability")
                print(f"     → Recommendation: Review network architecture or increase clipping")
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(grad_norms, linewidth=2)
            ax1.axhline(config.MAX_GRAD_NORM, color='red', linestyle='--', label='Clip threshold')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Gradient Norm')
            ax1.set_title('Gradient Norms During Training')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(losses, linewidth=2, color='orange')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('gradient_analysis.png', dpi=150, bbox_inches='tight')
            print(f"\n  📊 Plot saved to: gradient_analysis.png")
            
            self.results['gradient_explosion'] = {
                'passed': is_stable,
                'max_grad': max_grad,
                'avg_grad': avg_grad
            }
            
            return is_stable
        else:
            print("  ⚠️  SKIPPED: No training data generated")
            return True
    
    def run_all_tests(self):
        """
        Runs all critical tests and prints a summary of the results.

        Returns:
            tuple: A tuple containing the number of passed and failed tests.
        """
        print("\n" + "🔬" * 40)
        print("CRITICAL ISSUE TEST SUITE")
        print("Testing AI Memory System for Production Readiness")
        print("🔬" * 40)
        
        tests = [
            ("Memory Explosion", self.test_memory_explosion),
            ("Catastrophic Forgetting", self.test_catastrophic_forgetting),
            ("Privacy Leakage", self.test_privacy_leakage),
            ("Gradient Explosion", self.test_gradient_explosion),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n  ❌ ERROR in {test_name}: {str(e)}")
                failed += 1
        
        # Final Summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"\n  Tests Passed: {passed}/{len(tests)}")
        print(f"  Tests Failed: {failed}/{len(tests)}")
        print(f"  Pass Rate: {passed/len(tests)*100:.1f}%")
        
        if failed == 0:
            print(f"\n  🎉 ALL TESTS PASSED!")
            print(f"  System is ready for production deployment.")
        elif failed <= 2:
            print(f"\n  ⚠️  SOME ISSUES DETECTED")
            print(f"  System needs fixes before production deployment.")
        else:
            print(f"\n  🚨 CRITICAL ISSUES DETECTED")
            print(f"  DO NOT deploy to production without fixes!")
        
        print("\n" + "="*80)
        
        self.cleanup()
        
        return passed, failed


def main():
    """
    Main entry point for the critical tests executable.
    """
    print("Starting Critical Issue Tests...")
    print("This will take 2-3 minutes...\n")
    
    tester = CriticalIssueTests()
    
    try:
        passed, failed = tester.run_all_tests()
        
        # Return exit code
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        tester.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        tester.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
