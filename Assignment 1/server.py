"""
server.py
Central server for federated learning
Aggregates client models using Federated Averaging (FedAvg)
"""
import numpy as np
from model import LinearModel


class Server:
    """
    Federated Learning Server
    Coordinates training and aggregates client model updates
    """
    
    def __init__(self):
        """Initialize server with global model"""
        self.global_model = LinearModel()
        self.history = {
            'rounds': [],
            'weights': [],
            'biases': [],
            'avg_loss': []
        }
    
    def get_global_params(self):
        """
        Get current global model parameters
        
        Returns:
            Tuple of (weight, bias)
        """
        return self.global_model.get_params()
    
    def aggregate(self, client_params, method='average'):
        """
        Aggregate client model parameters using FedAvg
        
        Args:
            client_params: List of (weight, bias) tuples from clients
            method: Aggregation method ('average' or 'weighted')
        
        Returns:
            None (updates global model in place)
        """
        if method == 'average':
            # Simple average (FedAvg)
            avg_w = np.mean([params[0] for params in client_params])
            avg_b = np.mean([params[1] for params in client_params])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Update global model
        self.global_model.set_params(avg_w, avg_b)
    
    def record_round(self, round_num, avg_loss=None):
        """
        Record training history
        
        Args:
            round_num: Current round number
            avg_loss: Average loss across clients (optional)
        """
        w, b = self.global_model.get_params()
        self.history['rounds'].append(round_num)
        self.history['weights'].append(w)
        self.history['biases'].append(b)
        if avg_loss is not None:
            self.history['avg_loss'].append(avg_loss)
    
    def get_history(self):
        """
        Get training history
        
        Returns:
            Dictionary with training metrics over rounds
        """
        return self.history
    
    def print_status(self, round_num):
        """
        Print current global model status
        
        Args:
            round_num: Current round number
        """
        w, b = self.global_model.get_params()
        print(f"Round {round_num}: w = {w:.4f}, b = {b:.4f}")