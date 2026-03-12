"""
Federated Learning - Client Side
- Local model training on private data
- Model update (gradient/weight diff) transmission to central server
"""

import numpy as np
import requests
import json
import copy
import logging

logging.basicConfig(level=logging.INFO, format="[CLIENT %(asctime)s] %(message)s")

# ─────────────────────────────────────────────
# Simple Neural Network (NumPy only, no torch needed)
# ─────────────────────────────────────────────
class SimpleNN:
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        self.params = {
            "W1": np.random.randn(input_dim, hidden_dim) * 0.01,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, output_dim) * 0.01,
            "b2": np.zeros(output_dim),
        }

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.params["W1"] + self.params["b1"]
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.params["W2"] + self.params["b2"]
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        log_probs = -np.log(y_pred[np.arange(n), y_true] + 1e-9)
        return log_probs.mean()

    def backward(self, y_true, lr=0.01):
        n = y_true.shape[0]
        grads = {}

        # Output layer gradient
        dz2 = self.a2.copy()
        dz2[np.arange(n), y_true] -= 1
        dz2 /= n

        grads["W2"] = self.a1.T @ dz2
        grads["b2"] = dz2.sum(axis=0)

        # Hidden layer gradient
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (self.z1 > 0)

        grads["W1"] = self.X.T @ dz1
        grads["b1"] = dz1.sum(axis=0)

        # Update params
        for k in self.params:
            self.params[k] -= lr * grads[k]

        return grads

    def set_params(self, params: dict):
        """Load global model weights from server."""
        self.params = {k: np.array(v) for k, v in params.items()}

    def get_params(self) -> dict:
        return {k: v.tolist() for k, v in self.params.items()}


# ─────────────────────────────────────────────
# Federated Learning Client
# ─────────────────────────────────────────────
class FLClient:
    def __init__(self, client_id: str, server_url: str, local_epochs: int = 5, lr: float = 0.01):
        self.client_id = client_id
        self.server_url = server_url
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = SimpleNN()

    # ── Step 1: Pull global model from server ──────────────────────────────
    def pull_global_model(self):
        logging.info(f"[{self.client_id}] Pulling global model from server...")
        try:
            resp = requests.get(f"{self.server_url}/get_model", timeout=10)
            resp.raise_for_status()
            global_params = resp.json()["params"]
            self.model.set_params(global_params)
            logging.info(f"[{self.client_id}] Global model loaded.")
        except requests.exceptions.ConnectionError:
            logging.warning(f"[{self.client_id}] Server not reachable. Using local init.")

    # ── Step 2: Local training on private data ─────────────────────────────
    def local_train(self, X: np.ndarray, y: np.ndarray):
        logging.info(f"[{self.client_id}] Starting local training for {self.local_epochs} epochs...")
        initial_params = copy.deepcopy(self.model.params)

        for epoch in range(self.local_epochs):
            y_pred = self.model.forward(X)
            loss = self.model.compute_loss(y_pred, y)
            self.model.backward(y, lr=self.lr)
            if (epoch + 1) % max(1, self.local_epochs // 3) == 0:
                logging.info(f"[{self.client_id}] Epoch {epoch+1}/{self.local_epochs} | Loss: {loss:.4f}")

        # Compute weight update (delta) = trained_weights - initial_weights
        weight_delta = {
            k: (self.model.params[k] - initial_params[k]).tolist()
            for k in self.model.params
        }
        logging.info(f"[{self.client_id}] Local training complete.")
        return weight_delta

    # ── Step 3: Send update to server ─────────────────────────────────────
    def send_update(self, weight_delta: dict, num_samples: int):
        payload = {
            "client_id": self.client_id,
            "num_samples": num_samples,
            "weight_delta": weight_delta,
            "full_weights": self.model.get_params(),  # Also send full weights (optional)
        }
        logging.info(f"[{self.client_id}] Sending model update to server ({num_samples} samples)...")
        try:
            resp = requests.post(
                f"{self.server_url}/submit_update",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            logging.info(f"[{self.client_id}] Server response: {result['message']}")
            return True
        except requests.exceptions.ConnectionError:
            logging.warning(f"[{self.client_id}] Could not reach server. Update not sent.")
            return False

    # ── Full federated round ───────────────────────────────────────────────
    def run_round(self, X: np.ndarray, y: np.ndarray):
        self.pull_global_model()
        weight_delta = self.local_train(X, y)
        success = self.send_update(weight_delta, num_samples=len(X))
        return success


# ─────────────────────────────────────────────
# Demo: Simulate local data and run one FL round
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic local dataset (private, never leaves this client)
    np.random.seed(42)
    X_local = np.random.randn(100, 4).astype(np.float32)
    y_local = (X_local[:, 0] + X_local[:, 1] > 0).astype(int)  # Binary labels

    client = FLClient(
        client_id="client_001",
        server_url="http://localhost:5000",  # Change to your server address
        local_epochs=10,
        lr=0.01,
    )

    logging.info("=" * 50)
    logging.info("Starting Federated Learning Round")
    logging.info("=" * 50)
    client.run_round(X_local, y_local)
