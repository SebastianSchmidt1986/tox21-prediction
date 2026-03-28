# GNNs vs Traditional ML for Tox21 Toxicity Prediction

Comparative study of Graph Neural Networks (GCN, GIN) against fingerprint-based baselines (Random Forest, XGBoost) for multi-task binary toxicity prediction on the [Tox21 dataset](https://tripod.nih.gov/tox21/challenge/).

**12,707 compounds · 12 qHTS assays · scaffold-based splitting · masked BCE loss**

---

## Overview

| | Baselines | GNNs |
|---|---|---|
| Input | Morgan fingerprints (ECFP4, 2048 bits) | Molecular graphs (node + edge features) |
| Models | Random Forest, XGBoost (per-task) | GCN, GIN (multi-task shared head) |
| Interpretability | SHAP (TreeExplainer) | Integrated gradients (Captum) |

Primary metric: **PR-AUC** (average precision), given extreme class imbalance (~2–10% active rate per assay).

---

## Repository Structure

```
├── src/
│   ├── data/
│   │   ├── load.py         # Tox21 loading from SDF/SMILES
│   │   ├── sanitise.py     # RDKit SMILES standardisation
│   │   ├── featurise.py    # Morgan fingerprints + PyG graph construction
│   │   └── split.py        # Scaffold-based splitting (Murcko scaffolds)
│   ├── models/
│   │   ├── baseline.py     # Random Forest, XGBoost wrappers
│   │   ├── gnn.py          # GCN / GIN (PyTorch Geometric)
│   │   └── multitask.py    # Shared multi-task head with masked loss
│   ├── training/
│   │   ├── train.py        # Training loop (masked BCE, pos_weight, early stopping)
│   │   └── evaluate.py     # Per-assay PR-AUC, ROC-AUC, F1
│   ├── interpret/
│   │   ├── shap_explain.py # SHAP for fingerprint models
│   │   └── gnn_explain.py  # Integrated gradients for GNNs (Captum)
│   └── utils/
│       ├── chem.py         # RDKit helpers
│       ├── metrics.py      # PR-AUC, ROC-AUC, threshold-tuned F1
│       └── plotting.py     # ROC/PR curves, per-assay bar charts
├── configs/
│   ├── baseline.yaml       # RF/XGBoost hyperparameters
│   ├── gcn.yaml            # GCN architecture + training config
│   └── gin.yaml            # GIN architecture + training config
├── data/
│   └── raw/                # Place downloaded Tox21 files here (not tracked by git)
├── outputs/                # Created automatically during training (not tracked by git)
└── requirements.txt
```

---

## Setup

**Python 3.11+ required.**

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade build tools first
pip install -U pip setuptools wheel

# 3. Install numba/llvmlite as binary wheels (required before other deps)
pip install --only-binary=:all: numba llvmlite

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> **Note on PyTorch + PyG:** `torch` and `torch-geometric` may need version-specific install commands depending on your CUDA version. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the above fails.

---

## Data

Download the Tox21 dataset from the [NIH Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/) and place the files in `data/raw/`. The pipeline expects either SDF or SMILES format as provided by the challenge organisers.

```
data/
└── raw/
    ├── tox21_10k_data_all.sdf    # or equivalent download
    └── tox21_10k_data_all.csv
```

The `data/processed/`, `data/splits/`, and `outputs/` directories are created automatically when you run the pipeline.

---

## Running the Pipeline

### 1. Preprocessing

```bash
python -m src.data.sanitise    # SMILES sanitisation via RDKit
python -m src.data.featurise   # Morgan fingerprints + graph features
python -m src.data.split       # Scaffold-based 80/10/10 split
```

Outputs written to `data/processed/` and `data/splits/`.

### 2. Train Baselines

```bash
python -m src.training.train --config configs/baseline.yaml
```

Trains Random Forest and XGBoost per-task classifiers. Saved to `outputs/models/`.

### 3. Train GNNs

```bash
python -m src.training.train --config configs/gcn.yaml
python -m src.training.train --config configs/gin.yaml
```

Training uses masked BCE loss (missing labels are excluded, not treated as inactive) with per-task `pos_weight` to handle class imbalance. Early stopping is monitored on validation PR-AUC.

### 4. Evaluate

```bash
python -m src.training.evaluate --model outputs/models/random_forest.pkl
python -m src.training.evaluate --model outputs/models/gcn_best.pt
```

Outputs per-assay and mean PR-AUC / ROC-AUC / F1 to `outputs/results/`.

### 5. Interpretability

```bash
# SHAP for Random Forest / XGBoost
python -m src.interpret.shap_explain --model outputs/models/random_forest.pkl

# Integrated gradients for GCN
python -m src.interpret.gnn_explain --model outputs/models/gcn_best.pt
```

Figures saved to `outputs/figures/`.

---

## Design Decisions

- **Scaffold split** over random split — random splits inflate performance estimates because structurally similar molecules end up in both train and test.
- **Masked BCE** over label imputation — missing Tox21 labels are unobserved, not inactive. Treating them as negatives would introduce substantial noise.
- **`pos_weight`** over SMOTE — interpolating molecular fingerprints or graph features can produce chemically invalid samples.
- **PR-AUC as primary metric** — ROC-AUC is misleading under extreme class imbalance; average precision better reflects model utility for toxicity screening.

---

## Tests

```bash
pytest tests/ -v
```

Tests cover SMILES sanitisation edge cases, featurisation output shapes, scaffold split leakage, masked loss gradients, and metric functions.

---

## Linting

```bash
ruff check . && ruff format .
```

---

## Dependencies

Core: `rdkit`, `torch`, `torch-geometric`, `scikit-learn`, `xgboost`, `pandas`, `numpy`
Interpretability: `shap`, `captum`
Visualisation: `matplotlib`, `seaborn`
