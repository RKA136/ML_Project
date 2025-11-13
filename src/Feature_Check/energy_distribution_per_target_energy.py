import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import os

# ===========================
# Load processed data
# ===========================
with open("config.json", "r") as f:
    config = json.load(f)
data_dir = config["data_dir"]

data_path = os.path.join(data_dir, "processed_data_large_v1.pt")
data = torch.load(data_path)

X = data["X"].numpy()  # features: [E_sum, E_max, r_std, z_std, r90, E_layer_frac...]
y = data["y"].numpy().flatten()  # true energy

# ===========================
# Choose target energy
# ===========================
target_energy = 150.0  # GeV
tolerance = 10         # ± tolerance to select events

mask = (y >= target_energy - tolerance) & (y <= target_energy + tolerance)
E_sum_target = X[mask, 28]

# ===========================
# Fit histogram with Gaussian
# ===========================
mu, std = norm.fit(E_sum_target)  # fit mean and std

# Plot histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(E_sum_target, bins=100, density=True, color='skyblue', alpha=0.7, label='Data')

# Plot Gaussian fit
xmin, xmax = bins[0], bins[-1]
x_fit = np.linspace(xmin, xmax, 200)
y_fit = norm.pdf(x_fit, mu, std)
plt.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Gaussian Fit\nμ={mu:.2f}, σ={std:.2f}')

plt.xlabel("Total Deposited Energy (E_sum)")
plt.ylabel("Normalized Counts")
plt.title(f"Distribution of E_sum for Target Energy ≈ {target_energy} GeV")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
