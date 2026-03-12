"""
Federated Learning - Central Server
- Receives model updates from clients
- Aggregates using FedAvg (Federated Averaging)
- Broadcasts new global model to clients
"""

import numpy as np
import json
import logging
from flask import Flask, request, jsonify
from threading import Lock
import copy

logging.basicConfig(level=logging.INFO, format="[SERVER %(asctime)s] %(message)s")

app = Flask(__name__)

# ─────────────────────────────────────────────
# Global Model Store
# ─────────────────────────────────────────────
class GlobalModelStore:
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        # Initialize global model weights
        self.params = {
            "W1": (np.random.randn(input_dim, hidden_dim) * 0.01).tolist(),
            "b1": np.zeros(hidden_dim).tolist(),
            "W2": (np.random.randn(hidden_dim, output_dim) * 0.01).tolist(),
            "b2": np.zeros(output_dim).tolist(),
        }
        self.lock = Lock()
        self.pending_updates = []   # Buffer of client updates for aggregation
        self.round_number = 0
        self.min_clients = 2        # Minimum clients before aggregation

    def add_update(self, client_id: str, full_weights: dict, num_samples: int):
        with self.lock:
            self.pending_updates.append({
                "client_id": client_id,
                "weights": full_weights,
                "num_samples": num_samples,
            })
            logging.info(f"Received update from {client_id} ({num_samples} samples). "
                         f"Buffer: {len(self.pending_updates)}/{self.min_clients}")

            # Aggregate once enough clients have submitted
            if len(self.pending_updates) >= self.min_clients:
                self._fedavg_aggregate()

    def _fedavg_aggregate(self):
        """FedAvg: weighted average of client weights by number of samples."""
        updates = self.pending_updates
        total_samples = sum(u["num_samples"] for u in updates)

        logging.info(f"Aggregating {len(updates)} client updates "
                     f"(total samples: {total_samples}) — Round {self.round_number + 1}")

        new_params = {}
        for key in updates[0]["weights"]:
            # Weighted average
            aggregated = np.sum(
                [
                    np.array(u["weights"][key]) * (u["num_samples"] / total_samples)
                    for u in updates
                ],
                axis=0,
            )
            new_params[key] = aggregated.tolist()

        self.params = new_params
        self.round_number += 1
        self.pending_updates.clear()   # Reset buffer for next round

        logging.info(f"✅ Global model updated — Round {self.round_number} complete.")

    def get_params(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.params)


model_store = GlobalModelStore()


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.route("/get_model", methods=["GET"])
def get_model():
    """Clients pull the current global model."""
    params = model_store.get_params()
    logging.info(f"Global model dispatched to client (Round {model_store.round_number})")
    return jsonify({
        "round": model_store.round_number,
        "params": params,
    })


@app.route("/submit_update", methods=["POST"])
def submit_update():
    """Clients push their local model updates."""
    data = request.json
    client_id = data.get("client_id", "unknown")
    num_samples = data.get("num_samples", 1)
    full_weights = data.get("full_weights")    # Full trained weights

    if full_weights is None:
        return jsonify({"error": "Missing full_weights"}), 400

    model_store.add_update(client_id, full_weights, num_samples)

    return jsonify({
        "message": f"Update from {client_id} received. "
                   f"Buffer: {len(model_store.pending_updates)}/{model_store.min_clients}",
        "round": model_store.round_number,
    })


@app.route("/status", methods=["GET"])
def status():
    """Server status and round info."""
    return jsonify({
        "round_number": model_store.round_number,
        "pending_updates": len(model_store.pending_updates),
        "min_clients_for_aggregation": model_store.min_clients,
    })


@app.route("/set_min_clients", methods=["POST"])
def set_min_clients():
    """Dynamically configure aggregation threshold."""
    n = request.json.get("min_clients", 2)
    model_store.min_clients = max(1, int(n))
    return jsonify({"message": f"min_clients set to {model_store.min_clients}"})


# ─────────────────────────────────────────────
# Simulation: Multi-client FL without HTTP
# (Run this if you don't want to spin up Flask)
# ─────────────────────────────────────────────
def simulate_federated_learning(num_clients=3, num_rounds=5):
    """Simulate FL end-to-end in one script."""
    print("\n" + "="*60)
    print("   FEDERATED LEARNING SIMULATION (No HTTP)")
    print("="*60)

    store = GlobalModelStore()

    for fl_round in range(1, num_rounds + 1):
        print(f"\n── Round {fl_round} ──────────────────────────────────────")
        updates = []

        for cid in range(num_clients):
            # Each client gets the global model
            global_params = store.get_params()
            global_weights = {k: np.array(v) for k, v in global_params.items()}

            # Simulate local dataset (private, stays on client)
            np.random.seed(fl_round * 100 + cid)
            n_samples = np.random.randint(50, 200)
            X = np.random.randn(n_samples, 4)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)

            # Local training (5 SGD steps)
            local_weights = copy.deepcopy(global_weights)
            lr = 0.01
            for _ in range(5):
                z1 = X @ local_weights["W1"] + local_weights["b1"]
                a1 = np.maximum(0, z1)
                z2 = a1 @ local_weights["W2"] + local_weights["b2"]
                e = np.exp(z2 - z2.max(axis=1, keepdims=True))
                a2 = e / e.sum(axis=1, keepdims=True)

                dz2 = a2.copy(); dz2[np.arange(n_samples), y] -= 1; dz2 /= n_samples
                local_weights["W2"] -= lr * (a1.T @ dz2)
                local_weights["b2"] -= lr * dz2.sum(0)
                da1 = dz2 @ local_weights["W2"].T
                dz1 = da1 * (z1 > 0)
                local_weights["W1"] -= lr * (X.T @ dz1)
                local_weights["b1"] -= lr * dz1.sum(0)

            # Compute loss
            z1 = X @ local_weights["W1"] + local_weights["b1"]
            a1 = np.maximum(0, z1)
            z2 = a1 @ local_weights["W2"] + local_weights["b2"]
            e = np.exp(z2 - z2.max(axis=1, keepdims=True))
            a2 = e / e.sum(axis=1, keepdims=True)
            loss = -np.log(a2[np.arange(n_samples), y] + 1e-9).mean()

            print(f"  Client {cid+1:02d} | samples={n_samples:3d} | loss={loss:.4f}")
            updates.append({"weights": {k: v.tolist() for k, v in local_weights.items()},
                            "num_samples": n_samples})

        # FedAvg aggregation on server
        total = sum(u["num_samples"] for u in updates)
        new_params = {}
        for key in updates[0]["weights"]:
            new_params[key] = np.sum(
                [np.array(u["weights"][key]) * (u["num_samples"] / total) for u in updates],
                axis=0
            ).tolist()
        store.params = new_params
        store.round_number = fl_round
        print(f"  ✅ Server aggregated {num_clients} clients (FedAvg, total={total} samples)")

    print("\n" + "="*60)
    print(f"   Training complete — {num_rounds} rounds, {num_clients} clients")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys

    if "--simulate" in sys.argv:
        # Pure simulation mode (no Flask needed)
        simulate_federated_learning(num_clients=4, num_rounds=5)
    else:
        # Start Flask server
        logging.info("Starting Federated Learning Server on http://0.0.0.0:5000")
        logging.info("Endpoints: GET /get_model | POST /submit_update | GET /status")
        app.run(host="0.0.0.0", port=5000, debug=False)
