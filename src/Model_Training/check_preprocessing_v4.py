#!/usr/bin/env python3
"""
This script loads processed calorimeter data stored in a PyTorch .pt file,
extracts per-layer energy features (E_sum, E1/E7, and E7/E19), and visualizes
the layer-wise energy ratio distributions for a selected event.

Steps performed:
1. Load the dataset configuration from config.json to locate the data directory.
2. Load the preprocessed dataset (processed_data_0001_v4.pt) using torch.load.
3. Extract feature tensors (X) and corresponding target energies (y).
4. Split the feature matrix X into three equal parts:
      - E_sum   → total energy deposit per layer
      - E1/E7   → energy ratio feature between core and surrounding region
      - E7/E19  → energy ratio feature between inner and outer regions
5. Print dataset shapes and display energy features for one example event.
6. Plot E1/E7 and E7/E19 ratio values across detector layers for visual inspection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json, os

with open("config.json") as f:
    data_dir = json.load(f)["data_dir"]

data = torch.load(os.path.join(data_dir, "processed_data_0001_v4.pt"), weights_only=True)
X = data["X"].numpy()
y = data["y"].numpy()

n_layers = X.shape[1] // 3
E_sum = X[:, :n_layers]
E1E7 = X[:, n_layers:2*n_layers]
E7E19 = X[:, 2*n_layers:]

print("X shape:", X.shape)
print("y shape:", y.shape)
print(f"n_layers = {n_layers}")

# Inspect one event
i = 0
print(f"\n=== Event {i} ===")
print("True energy:", y[i, 0])
print("Layer  |  E_sum   E1/E7   E7/E19")
for l in range(n_layers):
    print(f"{l:6d}  {E_sum[i,l]:8.3f}  {E1E7[i,l]:8.3f}  {E7E19[i,l]:8.3f}")

# Optional plot
plt.figure(figsize=(8,4))
plt.plot(E1E7[i], 'o-', label='E1/E7')
plt.plot(E7E19[i], 's-', label='E7/E19')
plt.title(f"Event {i}: Layer-wise energy ratios")
plt.xlabel("Layer index")
plt.ylabel("Ratio value")
plt.legend()
plt.grid(alpha=0.4)
plt.show()
