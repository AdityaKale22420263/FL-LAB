"""
client.py
Client implementation for federated learning
Each client has private data and trains locally
"""
import numpy as np
from model import LinearModel


class Client:
    """
    Federated Learning Client
    Holds private data and performs local training
    """
    
    def __init__(self, client_id, x, y):
        """
        Initialize client with private dataset
        
        Args:
            client_id: Unique identifier for this client
            x: Private input features
            y: Private labels
        """
        self.client_id = client_id
        self.x = x
        self.y = y
        self.model = LinearModel()
        
    def train(self, global_w, global_b, lr=0.01, epochs=50):
        """
        Train model locally on private data using gradient descent
        
        Args:
            global_w: Global weight from server
            global_b: Global bias from server
            lr: Learning rate for gradient descent
            epochs: Number of local training epochs
        
        Returns:
            Tuple of updated (weight, bias) parameters
        """
        # Set model to global parameters
        self.model.set_params(global_w, global_b)
        
        # Local training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model.predict(self.x)
            
            # Compute gradients
            dw = np.mean((y_pred - self.y) * self.x)
            db = np.mean(y_pred - self.y)
            
            # Update parameters using gradient descent
            self.model.w -= lr * dw
            self.model.b -= lr * db
        
        # Return updated parameters (NOT raw data)
        return self.model.get_params()
    
    def evaluate(self):
        """
        Evaluate current model on local data
        
        Returns:
            MSE loss on local dataset
        """
        return self.model.compute_loss(self.x, self.y)
    
    def get_data_size(self):
        """
        Get size of local dataset
        
        Returns:
            Number of samples
        """
        return len(self.x)