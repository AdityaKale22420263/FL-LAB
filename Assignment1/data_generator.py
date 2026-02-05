"""
data_generator.py
Generate synthetic datasets for federated learning clients
Simulates data heterogeneity across clients
"""
import numpy as np


def generate_linear_data(num_samples, true_w=2.0, true_b=1.0, noise_std=0.1):
    """
    Generate synthetic linear regression data: y = true_w * x + true_b + noise
    
    Args:
        num_samples: Number of data points
        true_w: True weight parameter
        true_b: True bias parameter
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Tuple of (x, y) numpy arrays
    """
    x = np.random.rand(num_samples)
    y = true_w * x + true_b + np.random.randn(num_samples) * noise_std
    return x, y


def create_client_datasets(num_clients, samples_per_client=50, 
                          true_w=2.0, true_b=1.0, 
                          heterogeneous=False):
    """
    Create datasets for multiple federated learning clients
    
    Args:
        num_clients: Number of clients to create data for
        samples_per_client: Number of samples each client gets
        true_w: True weight for data generation
        true_b: True bias for data generation
        heterogeneous: If True, add client-specific variations
    
    Returns:
        List of (x, y) tuples, one per client
    """
    datasets = []
    
    for i in range(num_clients):
        if heterogeneous:
            # Each client has slightly different data distribution
            client_w = true_w + np.random.randn() * 0.3
            client_b = true_b + np.random.randn() * 0.2
            noise = 0.1 + np.random.rand() * 0.1
        else:
            # All clients have same distribution
            client_w = true_w
            client_b = true_b
            noise = 0.1
        
        x, y = generate_linear_data(
            samples_per_client, 
            true_w=client_w, 
            true_b=client_b, 
            noise_std=noise
        )
        datasets.append((x, y))
    
    return datasets


def print_dataset_info(datasets):
    """
    Print information about generated datasets
    
    Args:
        datasets: List of (x, y) tuples
    """
    print(f"\n{'='*60}")
    print(f"Dataset Information")
    print(f"{'='*60}")
    print(f"Number of clients: {len(datasets)}")
    print(f"Samples per client: {len(datasets[0][0])}")
    print(f"Total samples: {sum(len(x) for x, _ in datasets)}")
    print(f"{'='*60}\n")