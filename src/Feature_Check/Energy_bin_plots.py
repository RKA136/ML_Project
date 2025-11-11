#!/usr/bin/env python3
"""
energy_resolution_analysis.py
--------------------------------------------------
Performs energy response and resolution analysis from raw calorimeter data.

Overview:
---------
1. Loads raw calorimeter data from HDF5 file.
2. Computes total measured energy per event.
3. Bins events in true energy (default: 10 bins).
4. For each bin:
     - Plots the measured energy distribution.
     - Fits a Gaussian to obtain mean (μ) and sigma (σ).
5. Plots:
     (a) Mean measured energy vs mean true energy
     (b) σ_E / μ_E vs true energy (energy resolution curve)

Outputs:
--------
- `energy_response_binned.png`
- `energy_resolution.png`
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import norm
from tqdm import tqdm

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
with open("config.json", "r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
figures_dir = config["figures_dir"]
input_file = os.path.join(data_dir, "hgcal_electron_data_large.h5")

n_bins = 10  # number of true energy bins
save_histograms = True  # whether to save per-bin Gaussian fit plots

# ------------------------------------------------------------
# Load raw data
# ------------------------------------------------------------
print(f"Loading data from {input_file} ...")
with h5py.File(input_file, "r") as f:
    nhits = np.array(f["nhits"], dtype=np.int32)
    true_energy = np.array(f["true_energy"], dtype=np.float32)
    rechit_energy = np.array(f["rechit_energy"], dtype=np.float32)

# ------------------------------------------------------------
# Compute per-event measured energy
# ------------------------------------------------------------
print("Computing total measured energy per event...")
measured_energy = np.zeros_like(true_energy)
offsets = np.cumsum(np.insert(nhits, 0, 0))

for i in range(len(nhits)):
    start = offsets[i]
    end = offsets[i + 1]
    measured_energy[i] = np.sum(rechit_energy[start:end])

# ------------------------------------------------------------
# Bin events by true energy
# ------------------------------------------------------------
bin_edges = np.linspace(np.min(true_energy), np.max(true_energy), n_bins + 1)
bin_indices = np.digitize(true_energy, bin_edges) - 1  # 0-based bin indices

mean_true = []
mean_measured = []
sigma_over_mean = []

print("Processing bins and fitting Gaussian distributions...")
for b in tqdm(range(n_bins)):
    mask = bin_indices == b
    if not np.any(mask):
        continue

    E_true_bin = true_energy[mask]
    E_meas_bin = measured_energy[mask]

    mu, sigma = norm.fit(E_meas_bin)
    mean_true.append(np.mean(E_true_bin))
    mean_measured.append(mu)
    sigma_over_mean.append(sigma / mu)

    # Plot the Gaussian fit for this bin
    if save_histograms:
        plt.figure(figsize=(6, 4))
        n, bins, _ = plt.hist(E_meas_bin, bins=50, density=True, alpha=0.6, label="Measured E")
        x = np.linspace(min(bins), max(bins), 500)
        plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f"Fit: μ={mu:.2f}, σ={sigma:.2f}")
        plt.xlabel("Measured Energy (GeV)")
        plt.ylabel("Probability Density")
        plt.title(f"True Energy ≈ {np.mean(E_true_bin):.2f} GeV")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"Energy_bin_plots/hist_fit_bin_{b+1}.png"), dpi=200)
        plt.close()

mean_true = np.array(mean_true)
mean_measured = np.array(mean_measured)
sigma_over_mean = np.array(sigma_over_mean)

# ------------------------------------------------------------
# Plot Mean Response
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.plot(mean_true, mean_measured, 'o-', lw=2, label="Measured Response")
plt.plot(mean_true, mean_true, 'k--', label="Ideal Response (y=x)")
plt.xlabel("Mean True Energy (GeV)", fontsize=13)
plt.ylabel("Mean Measured Energy (GeV)", fontsize=13)
plt.title(f"Calorimeter Energy Response ({n_bins} bins)", fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Energy_bin_plots/energy_response_binned.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot Energy Resolution σ_E / ⟨E⟩
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.plot(mean_true, sigma_over_mean, 'o-', lw=2)
plt.xlabel("True Energy (GeV)", fontsize=13)
plt.ylabel("σ_E / ⟨E⟩", fontsize=13)
plt.title("Calorimeter Energy Resolution", fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Energy_bin_plots/energy_resolution.png"), dpi=300)
plt.show()

print("\nAnalysis complete.")
print(f"- Response curve saved to: {os.path.join(figures_dir, 'Energy_bin_plots/energy_response_binned.png')}")
print(f"- Resolution curve saved to: {os.path.join(figures_dir, 'Energy_bin_plots/energy_resolution.png')}")
if save_histograms:
    print(f"- Per-bin Gaussian fits saved to {figures_dir}/Energy_bin_plots/hist_fit_bin_*.png")
