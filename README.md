# Federated Learning Assignments

A collection of assignments exploring Federated Learning concepts, implementations, and applications on real-world datasets.

## Assignments

### Assignment 1 — Basic Federated Learning Setup
Client-server federated learning architecture with custom model training and aggregation.
- `client.py` — Client-side local training
- `server.py` — Server-side model aggregation
- `main.py` — Entry point to run the FL pipeline

### Assignment 2 — Data Preprocessing & Federated Distribution
Preprocessing and partitioning a student performance dataset across 5 federated clients.
- `codes/generate_dataset.py` — Generates partitioned client datasets
- `codes/federated_learning_preprocessing.py` — Preprocessing utilities
- `federated_data/` — Partitioned CSV files for each client

### Assignment 3 — Federated Learning on Housing Dataset
Notebook-based FL analysis using a housing dataset.
- `FL3.ipynb` — Main notebook

### Assignment 4 — Simplified FL Client-Server
Minimal federated learning implementation with a lightweight client-server setup.
- `fl_client.py` — FL client
- `fl_server.py` — FL server

### Assignment 5 — Federated Learning on MNIST & Diabetes
FL applied to image classification (MNIST) and tabular data (diabetes).
- `ass_5.ipynb` — Main notebook

### Assignment 6 — Vertical Federated Learning on Heart Disease
Vertical FL where two clients hold different features of the heart disease dataset, with a server aggregating embeddings.
- `pr6.ipynb` — Main notebook
- `heart.csv` — Heart disease dataset

## Tech Stack
- Python, PyTorch, scikit-learn, pandas, NumPy
- Jupyter Notebooks

## How to Run
1. Install dependencies:
   ```
   pip install torch scikit-learn pandas numpy
   ```
2. Navigate to an assignment folder and run the script or open the notebook.
