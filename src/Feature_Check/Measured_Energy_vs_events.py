#!/usr/bin/env python3
"""
energy_slice_bincount_gaussian.py
-------------------------------------------------------
Analyzes reconstructed (measured) energy distribution for
events near a chosen true energy value using bin counting
and overlays a Gaussian fit.

Workflow:
---------
1. Loads raw calorimeter event data.
2. Computes total measured energy per event.
3. Selects events within a tolerance window around a target true energy.
4. Uses `np.histogram` to compute bin counts.
5. Fits a Gaussian to the measured energy distribution.
6. Plots bin centers (counts) and the Gaussian fit curve.

Usage:
------
    python energy_slice_bincount_gaussian.py
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import json
import os
from scipy.stats import norm

# ------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------
with open("config.json", "r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
figures_dir = config["figures_dir"]
os.makedirs(os.path.join(figures_dir, "Energy_counts"), exist_ok=True)

input_file = os.path.join(data_dir, "hgcal_electron_data_large.h5")

# ------------------------------------------------------------
# Load raw calorimeter data
# ------------------------------------------------------------
print(f"Loading data from {input_file} ...")
with h5py.File(input_file, "r") as f:
    nhits = np.array(f["nhits"], dtype=np.int32)
    true_energy = np.array(f["true_energy"], dtype=np.float32)
    rechit_energy = np.array(f["rechit_energy"], dtype=np.float32)

# ------------------------------------------------------------
# Compute total measured energy per event
# ------------------------------------------------------------
print("Computing total measured energy per event...")
measured_energy = np.zeros_like(true_energy)
offsets = np.cumsum(np.insert(nhits, 0, 0))

for i in range(len(nhits)):
    start = offsets[i]
    end = offsets[i + 1]
    measured_energy[i] = np.sum(rechit_energy[start:end])

# ------------------------------------------------------------
# Select events around a target true energy
# ------------------------------------------------------------
true_energy_target = 300.0  # GeV
tolerance = 0.5             # GeV window
mask = np.abs(true_energy - true_energy_target) <= tolerance
selected_measured = measured_energy[mask]

print(f"Selected {len(selected_measured)} events around {true_energy_target} ± {tolerance} GeV.")
if len(selected_measured) == 0:
    raise RuntimeError("No events found within the specified tolerance window.")

# ------------------------------------------------------------
# Compute bin counts (no histogram plotting)
# ------------------------------------------------------------
n_bins = 60
counts, bin_edges = np.histogram(selected_measured, bins=n_bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ------------------------------------------------------------
# Fit a Gaussian to measured energy distribution
# ------------------------------------------------------------
mu, sigma = norm.fit(selected_measured)
print(f"Gaussian fit parameters: μ = {mu:.3f} GeV, σ = {sigma:.3f} GeV")

# Generate smooth Gaussian curve scaled to match histogram counts
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
pdf_fit = norm.pdf(x_fit, mu, sigma)
pdf_fit_scaled = pdf_fit * (np.max(counts) / np.max(pdf_fit))  # scale to histogram height

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(x_fit, pdf_fit_scaled, "r-", lw=2.0, label=f"Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}")
plt.scatter(bin_centers, counts, s=15, color="black", marker="o", label="Binned Counts")

plt.xlabel("Measured Energy (GeV)", fontsize=13)
plt.ylabel("Event Count per Bin", fontsize=13)
plt.title(f"Measured Energy Distribution (True ≈ {true_energy_target:.1f} ± {tolerance:.1f} GeV)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
output_dir = os.path.join(figures_dir, "Energy_counts")
output_path = os.path.join(output_dir, f"bincount_gaussian_{int(true_energy_target)}GeV.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved to {output_path}")
