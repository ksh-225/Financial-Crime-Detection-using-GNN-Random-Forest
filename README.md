# Financial Crime Detection — GNN + Random Forest

Detecting illicit Bitcoin transactions using a hybrid **Random Forest + Graph Convolutional Network** approach on the Elliptic dataset.

## Overview

Traditional ML models treat transactions in isolation. This project models the Bitcoin transaction network as a graph and combines:
- **Random Forest** — captures feature-level fraud patterns
- **Graph Convolutional Network (GCN)** — captures relational patterns between transactions

The RF fraud probability is appended as an extra node feature before training the GCN, giving the model both local and structural signals.

## Result

| Model | F1 Score |
|---|---|
| Random Forest only | 0.75 – 0.82 |
| GNN only | 0.70 – 0.80 |
| **RF + GCN (this work)** | **0.87** |

## Dataset

[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — download and place these 3 files in the repo root:
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

> 203,769 transactions · 234,355 edges · 166 features/node · ~2% illicit

## Setup

```bash
pip install torch torch-geometric scikit-learn pandas numpy
```

Then open `financial_crime_detection.ipynb` and run all cells.

## Pipeline

1. Load and preprocess data
2. Train Random Forest on labeled transactions
3. Append RF fraud probability to node features
4. Construct transaction graph
5. Train 2-layer GCN
6. Evaluate using F1 score

## Why F1?

The dataset is highly imbalanced (~2% illicit). Accuracy would be misleading — F1 balances precision and recall for the minority class.

## References

- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with GCNs for Financial Forensics*, KDD 2019
- Alarab & Prakoonwit, *Robust Recurrent GCN for Illicit Transaction Prediction*, Multimedia Tools and Applications, 2024
