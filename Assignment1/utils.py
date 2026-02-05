"""
utils.py
Utility functions for visualization and analysis
"""
import numpy as np


def plot_training_history(server):
    """
    Plot training history (requires matplotlib)
    
    Args:
        server: Server object with training history
    """
    try:
        import matplotlib.pyplot as plt
        
        history = server.get_history()
        rounds = history['rounds']
        weights = history['weights']
        biases = history['biases']
        losses = history['avg_loss']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Weight convergence
        axes[0].plot(rounds, weights, marker='o', linewidth=2)
        axes[0].axhline(y=2.0, color='r', linestyle='--', label='True value')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Weight (w)')
        axes[0].set_title('Weight Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Bias convergence
        axes[1].plot(rounds, biases, marker='o', linewidth=2, color='orange')
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='True value')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Bias (b)')
        axes[1].set_title('Bias Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Loss over rounds
        axes[2].plot(rounds, losses, marker='o', linewidth=2, color='green')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Average Loss (MSE)')
        axes[2].set_title('Training Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fl_training_history.png', dpi=150)
        print("\n✓ Training history plot saved as 'fl_training_history.png'")
        plt.show()
        
    except ImportError:
        print("\n⚠ matplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")


def compare_federated_vs_centralized(datasets, num_rounds=10):
    """
    Compare federated learning vs centralized training
    
    Args:
        datasets: List of client datasets
        num_rounds: Number of federated rounds
    """
    from client import Client
    from server import Server
    
    print("\n" + "="*60)
    print("COMPARISON: Federated vs Centralized Learning")
    print("="*60)
    
    # Federated Learning
    print("\n[1] Federated Learning (Privacy-Preserving)")
    clients = [Client(i, x, y) for i, (x, y) in enumerate(datasets)]
    server_fl = Server()
    
    for round_num in range(1, num_rounds + 1):
        global_w, global_b = server_fl.get_global_params()
        client_params = [c.train(global_w, global_b) for c in clients]
        server_fl.aggregate(client_params)
    
    w_fl, b_fl = server_fl.get_global_params()
    print(f"  Final model: w={w_fl:.4f}, b={b_fl:.4f}")
    
    # Centralized Learning (violates privacy)
    print("\n[2] Centralized Learning (Privacy-Violating)")
    all_x = np.concatenate([x for x, _ in datasets])
    all_y = np.concatenate([y for _, y in datasets])
    
    from model import LinearModel
    central_model = LinearModel()
    lr = 0.01
    
    for epoch in range(num_rounds * 50):
        y_pred = central_model.predict(all_x)
        dw = np.mean((y_pred - all_y) * all_x)
        db = np.mean(y_pred - all_y)
        central_model.w -= lr * dw
        central_model.b -= lr * db
    
    w_cent, b_cent = central_model.get_params()
    print(f"  Final model: w={w_cent:.4f}, b={b_cent:.4f}")
    
    print("\n[3] Comparison Results")
    print(f"  Difference in w: {abs(w_fl - w_cent):.6f}")
    print(f"  Difference in b: {abs(b_fl - b_cent):.6f}")
    print("\n✓ Federated learning achieves similar accuracy")
    print("✓ But preserves data privacy!")
    print("="*60 + "\n")


def print_algorithm_explanation():
    """Print explanation of FedAvg algorithm"""
    print("\n" + "="*60)
    print("FEDERATED AVERAGING (FedAvg) ALGORITHM")
    print("="*60)
    print("""
1. Server initializes global model θ₀

2. FOR each round t = 1, 2, ..., T:
   
   a) Server sends θₜ to all clients
   
   b) FOR each client k = 1, 2, ..., K (in parallel):
      - Client k receives θₜ
      - Client k trains on local data Dₖ
      - Client k computes updated parameters θₖ
      - Client k sends θₖ to server
   
   c) Server aggregates:
      θₜ₊₁ = (1/K) Σ θₖ  (average of client parameters)

3. Return final global model θₜ

KEY PROPERTIES:
• Privacy: Raw data never leaves clients
• Efficiency: Reduces communication overhead
• Scalability: Parallel client training
• Convergence: Provably converges for convex problems
    """)
    print("="*60 + "\n")