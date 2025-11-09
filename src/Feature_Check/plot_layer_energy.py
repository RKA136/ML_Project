import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load processed data
with open("config.json", "r") as f:
    config = json.load(f)
    data_dir = config["data_dir"]

data = torch.load(os.path.join(data_dir, "processed_data_large_v3.pt"))
X = data["X"].numpy()  # shape: (n_events, n_layers)
y = data["y"].numpy().flatten()  # shape: (n_events,)

# Specify the target beam energies you want to plot
target_values = [20.0, 100.0, 300.0]  # GeV
tolerance = 1.0  # ± tolerance around target value

plt.figure(figsize=(9,6))
plt.style.use("ggplot")
colors = ['blue', 'green', 'red']

for target, color in zip(target_values, colors):
    # Select events within tolerance
    mask = np.abs(y - target) <= tolerance
    X_selected = X[mask]

    if X_selected.shape[0] == 0:
        print(f"No events found around {target} GeV. Skipping.")
        continue

    # Compute mean and SEM per layer
    mean_frac = X_selected.mean(axis=0)
    sem_frac = X_selected.std(axis=0) / np.sqrt(X_selected.shape[0])

    n_layers = X_selected.shape[1]
    layers = np.arange(1, n_layers + 1)

    # Plot with error bars
    plt.errorbar(layers, mean_frac, yerr=sem_frac, fmt='o--', capsize=4,
                markersize=3, color=color, label=f"{target} GeV")

plt.xlabel("Layer Number")
plt.ylabel("Mean Measured Energy per Layer")
plt.title("Longitudinal Electromagnetic Shower Profile")
plt.xticks(layers)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()