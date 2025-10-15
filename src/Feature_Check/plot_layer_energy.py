import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ===========================
# Load processed data
# ===========================
with open("config.json", "r") as f:
    config = json.load(f)
data_dir = config["data_dir"]

data_path = os.path.join(data_dir, "processed_data_large.pt")
data = torch.load(data_path)

X = data["X"].numpy()  # features: [E_sum, E_max, r_std, z_std, r90, E_layer_frac...]
y = data["y"].numpy().flatten()  # true energy

# ===========================
# Target energies
# ===========================
target_energy_list = [20.0, 100.0, 300.0]  # GeV
tolerance = 0.5         # ± tolerance to select events
colors = ['skyblue', 'salmon', 'limegreen']

plt.figure(figsize=(10, 6))

for target_energy, color in zip(target_energy_list, colors):
    mask = (y >= target_energy - tolerance) & (y <= target_energy + tolerance)
    X_target = X[mask]

    if X_target.shape[0] == 0:
        print(f"No events found for target energy ≈ {target_energy} GeV")
        continue

    # Layer-wise absolute energy
    E_sum_target = X_target[:, 0]
    E_layer_frac = X_target[:, 5:]
    n_layers = E_layer_frac.shape[1]
    E_layer_abs = E_layer_frac * E_sum_target[:, None]

    # Mean and std per layer
    mean_E_layer = np.mean(E_layer_abs, axis=0)
    std_E_layer = np.std(E_layer_abs, axis=0) # Change the errorbar

    layers = np.arange(1, n_layers + 1)
    plt.errorbar(layers, mean_E_layer, yerr=std_E_layer, fmt='o-', 
                 color=color, ecolor='gray', elinewidth=1.5, capsize=3, alpha=0.8,
                 label=f'{target_energy} GeV')

plt.xlabel("Z Layer Index")
plt.ylabel("Measured Energy per Layer")
plt.title("Measured Energy per Layer for Multiple Target Energies")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
