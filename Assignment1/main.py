"""
main.py
Main script to run federated learning simulation

Assignment 1: Federated Learning
Demonstrates core FL workflow with FedAvg algorithm
"""
import numpy as np
from client import Client
from server import Server
from data_generator import create_client_datasets, print_dataset_info


def run_federated_learning(num_clients=5, num_rounds=10, 
                           samples_per_client=50,
                           local_epochs=50, learning_rate=0.01,
                           heterogeneous=False, seed=42):
    """
    Run complete federated learning simulation
    
    Args:
        num_clients: Number of participating clients
        num_rounds: Number of federated learning rounds
        samples_per_client: Dataset size for each client
        local_epochs: Number of local training epochs per round
        learning_rate: Learning rate for gradient descent
        heterogeneous: Whether to use heterogeneous data
        seed: Random seed for reproducibility
    
    Returns:
        Server object with training history
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    print("\n" + "="*60)
    print("FEDERATED LEARNING SIMULATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Samples per client: {samples_per_client}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data heterogeneity: {heterogeneous}")
    
    # Step 1: Generate datasets for clients (IID or non-IID)
    print("\n[Step 1] Generating client datasets...")
    datasets = create_client_datasets(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        heterogeneous=heterogeneous
    )
    print_dataset_info(datasets)
    
    # Step 2: Initialize clients with private data
    print("[Step 2] Initializing clients with private datasets...")
    clients = []
    for i, (x, y) in enumerate(datasets):
        client = Client(client_id=i, x=x, y=y)
        clients.append(client)
    print(f"✓ Created {len(clients)} clients\n")
    
    # Step 3: Initialize central server
    print("[Step 3] Initializing central server...")
    server = Server()
    w_init, b_init = server.get_global_params()
    print(f"✓ Server initialized with w={w_init:.4f}, b={b_init:.4f}\n")
    
    # Step 4: Federated training loop
    print("[Step 4] Starting federated training...\n")
    print("="*60)
    
    for round_num in range(1, num_rounds + 1):
        # Get current global parameters
        global_w, global_b = server.get_global_params()
        
        # List to store updated parameters from all clients
        client_params = []
        
        # Each client trains locally (in parallel in real FL)
        for client in clients:
            # Client receives global model and trains locally
            updated_params = client.train(
                global_w=global_w,
                global_b=global_b,
                lr=learning_rate,
                epochs=local_epochs
            )
            # Client sends only parameters (NOT raw data)
            client_params.append(updated_params)
        
        # Server aggregates client parameters using FedAvg
        server.aggregate(client_params, method='average')
        
        # Calculate average loss across clients
        avg_loss = np.mean([client.evaluate() for client in clients])
        
        # Record and display progress
        server.record_round(round_num, avg_loss)
        server.print_status(round_num)
        print(f"         Average Loss: {avg_loss:.6f}")
    
    print("="*60)
    print("\n[Step 5] Training completed!\n")
    
    # Final results
    final_w, final_b = server.get_global_params()
    print("Final Global Model:")
    print(f"  Weight (w): {final_w:.4f} (True value: 2.0)")
    print(f"  Bias (b):   {final_b:.4f} (True value: 1.0)")
    print(f"\nConvergence: Model learned y ≈ {final_w:.2f}x + {final_b:.2f}")
    print("="*60 + "\n")
    
    return server


def main():
    """Main entry point for federated learning simulation"""
    
    # Run simulation with default parameters
    server = run_federated_learning(
        num_clients=5,
        num_rounds=10,
        samples_per_client=50,
        local_epochs=50,
        learning_rate=0.01,
        heterogeneous=False,
        seed=42
    )
    
    # Optional: Demonstrate privacy preservation
    print("\n" + "="*60)
    print("PRIVACY PRESERVATION")
    print("="*60)
    print("\n✓ Raw data NEVER leaves client devices")
    print("✓ Only model parameters (w, b) are shared with server")
    print("✓ Server cannot reconstruct original client data")
    print("✓ Each client's dataset remains completely private")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()