"""
model.py
Linear regression model for federated learning simulation
"""
import numpy as np


class LinearModel:
    """
    Simple linear regression model: y = wx + b
    Used by both clients and server in federated learning
    """
    
    def __init__(self):
        """Initialize model with random weights"""
        self.w = np.random.randn()
        self.b = np.random.randn()
    
    def predict(self, x):
        """
        Make predictions using current model parameters
        
        Args:
            x: Input features (numpy array)
        
        Returns:
            Predicted values
        """
        return self.w * x + self.b
    
    def get_params(self):
        """
        Get current model parameters
        
        Returns:
            Tuple of (weight, bias)
        """
        return self.w, self.b
    
    def set_params(self, w, b):
        """
        Set model parameters
        
        Args:
            w: Weight parameter
            b: Bias parameter
        """
        self.w = w
        self.b = b
    
    def compute_loss(self, x, y):
        """
        Compute Mean Squared Error loss
        
        Args:
            x: Input features
            y: True labels
        
        Returns:
            MSE loss value
        """
        y_pred = self.predict(x)
        return np.mean((y_pred - y) ** 2)