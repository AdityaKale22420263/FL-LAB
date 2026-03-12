# Federated Learning Simulation
**Assignment 1: Federated Learning**

A Python implementation of basic Federated Learning with Federated Averaging (FedAvg) algorithm.

---

## 📁 Project Structure

```
federated-learning/
│
├── model.py              # Linear regression model
├── client.py             # Client implementation (local training)
├── server.py             # Server implementation (aggregation)
├── data_generator.py     # Dataset generation utilities
├── utils.py              # Visualization and analysis tools
├── main.py               # Main simulation script
└── README.md             # This file
```

---

## 🎯 Learning Objectives

This assignment demonstrates:

1. **Privacy-preserving distributed learning** - Raw data never leaves client devices
2. **Federated Averaging (FedAvg)** - Central aggregation of model parameters
3. **Client-Server architecture** - Communication pattern in federated systems
4. **Local training + Global aggregation** - Core FL workflow

---

## 🚀 Quick Start

### Installation

```bash
# No external dependencies required for basic simulation
python main.py
```

### Optional: For visualization

```bash
pip install matplotlib numpy
```

### Run Simulation

```bash
python main.py
```

---

## 📊 Expected Output

```
============================================================
FEDERATED LEARNING SIMULATION
============================================================

Configuration:
  Clients: 5
  Rounds: 10
  Samples per client: 50
  Local epochs: 50
  Learning rate: 0.01
  Data heterogeneity: False

[Step 1] Generating client datasets...
============================================================
Dataset Information
============================================================
Number of clients: 5
Samples per client: 50
Total samples: 250
============================================================

[Step 2] Initializing clients with private datasets...
✓ Created 5 clients

[Step 3] Initializing central server...
✓ Server initialized with w=-0.5432, b=0.8765

[Step 4] Starting federated training...

============================================================
Round 1: w = 1.8234, b = 0.9156
         Average Loss: 0.024531
Round 2: w = 1.9321, b = 0.9823
         Average Loss: 0.012456
Round 3: w = 1.9876, b = 1.0098
         Average Loss: 0.005678
...
Round 10: w = 2.0012, b = 0.9998
          Average Loss: 0.000234
============================================================

Final Global Model:
  Weight (w): 2.0012 (True value: 2.0)
  Bias (b):   0.9998 (True value: 1.0)

Convergence: Model learned y ≈ 2.00x + 1.00
============================================================
```

---

## 🧠 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CENTRAL SERVER                       │
│  • Maintains global model θ                            │
│  • Aggregates client updates (FedAvg)                  │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
     Sends θ │                          │ Receives updates
             ↓                          ↑
   ┌──────────────────────────────────────────────────┐
   │         CLIENT 1    CLIENT 2    ...   CLIENT K   │
   │  • Private data D₁    D₂      ...      Dₖ       │
   │  • Local training                               │
   │  • Sends parameters (NOT data)                  │
   └──────────────────────────────────────────────────┘
```

### Workflow

1. **Initialization**: Server creates initial global model
2. **Distribution**: Server sends model to all clients
3. **Local Training**: Each client trains on private data
4. **Aggregation**: Server averages client model parameters
5. **Update**: Global model updated with aggregated parameters
6. **Repeat**: Steps 2-5 for multiple rounds

### Federated Averaging (FedAvg)

```python
# Server aggregation
θ_global = (θ_client1 + θ_client2 + ... + θ_clientK) / K
```

---

## 🔬 Key Components

### 1. Model (`model.py`)

Simple linear regression: `y = wx + b`

- Forward pass: predictions
- Parameter management: get/set weights
- Loss computation: MSE

### 2. Client (`client.py`)

- Stores **private dataset** (never shared)
- Trains model locally using gradient descent
- Returns **only parameters** to server

### 3. Server (`server.py`)

- Maintains global model
- Aggregates client parameters (FedAvg)
- Tracks training history

### 4. Data Generator (`data_generator.py`)

- Creates synthetic datasets for clients
- Supports IID (identical distribution)
- Supports non-IID (heterogeneous data)

---

## 🧪 Experiments

### Experiment 1: Basic FL

```python
from main import run_federated_learning

server = run_federated_learning(
    num_clients=5,
    num_rounds=10,
    heterogeneous=False
)
```

### Experiment 2: Data Heterogeneity

```python
server = run_federated_learning(
    num_clients=10,
    num_rounds=20,
    heterogeneous=True  # Non-IID data
)
```

### Experiment 3: Visualization

```python
from utils import plot_training_history

plot_training_history(server)
```

### Experiment 4: FL vs Centralized

```python
from utils import compare_federated_vs_centralized
from data_generator import create_client_datasets

datasets = create_client_datasets(num_clients=5)
compare_federated_vs_centralized(datasets)
```

---

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Weight convergence** | How close w gets to true value (2.0) |
| **Bias convergence** | How close b gets to true value (1.0) |
| **Average loss** | MSE across all clients |
| **Rounds to convergence** | Number of rounds needed |

---

## 🔐 Privacy Guarantees

| Aspect | Status |
|--------|--------|
| Raw data sharing | ❌ Never shared |
| Model parameters | ✅ Shared (aggregated) |
| Client data size | ✅ Can be shared |
| Client identity | ⚠️ Known to server |

**Note**: This is basic FL. For stronger privacy, add:
- Differential Privacy (DP)
- Secure Aggregation
- Encrypted communication

---

## 🎓 Viva Questions & Answers

### Q1: What is Federated Learning?

**A**: Distributed machine learning where clients train models on local data without sharing raw data with a central server.

### Q2: What is FedAvg?

**A**: Federated Averaging - an algorithm that aggregates client model parameters by computing their average.

### Q3: Why is FL privacy-preserving?

**A**: Because raw data never leaves client devices. Only model parameters are shared.

### Q4: What are the limitations?

**A**: 
- Communication overhead
- Data heterogeneity challenges
- Privacy not perfect (model updates can leak info)
- Requires multiple training rounds

### Q5: Centralized vs Federated?

**A**: 
- **Centralized**: All data in one place, faster but privacy risk
- **Federated**: Data distributed, slower but privacy-preserving

---

## 📚 Theory Concepts

### Gradient Descent Update

```
w = w - α * ∂L/∂w
b = b - α * ∂L/∂b

where:
  α = learning rate
  L = loss function (MSE)
```

### Mean Squared Error (MSE)

```
MSE = (1/n) Σ(y_pred - y_true)²
```

### FedAvg Aggregation

```
θ_t+1 = (1/K) Σ θ_k

where:
  K = number of clients
  θ_k = parameters from client k
```

---

## 🛠️ Customization

### Change number of clients

```python
run_federated_learning(num_clients=10)
```

### Adjust learning parameters

```python
run_federated_learning(
    local_epochs=100,
    learning_rate=0.005
)
```

### Enable non-IID data

```python
run_federated_learning(heterogeneous=True)
```

---

## 📖 References

1. McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Kairouz et al. (2019) - "Advances and Open Problems in Federated Learning"

---

## 👨‍💻 Assignment Submission Checklist

- [x] Complete implementation of FL system
- [x] Client-server architecture
- [x] FedAvg aggregation algorithm
- [x] Privacy preservation (no raw data sharing)
- [x] Documentation and comments
- [x] Sample output demonstration
- [x] README with usage instructions

---

## 🎯 Core Takeaways

1. **FL enables collaborative learning** without data centralization
2. **FedAvg is simple yet effective** for parameter aggregation
3. **Privacy is preserved** but not perfect
4. **Trade-off**: Privacy vs Communication vs Accuracy

---

**Author**: [Your Name]  
**Course**: Federated Learning  
**Assignment**: Assignment 1 - Basic FL Simulation  
**Date**: January 2026