#!/usr/bin/env python3
"""
validate_model_mode4_gaussian_scan_v2.py
-------------------------------------------------------
Performs validation over multiple true energy values for a trained XGBoost model.

Workflow:
1. Loads trained model (mode4 + chosen target mode)
2. Iterates over selected true energies (e.g., 20, 30, ..., 350)
3. For each energy:
    - Selects events where true energy ≈ E ± tolerance
    - Predicts using the model
    - Converts predictions back to physical energy (GeV)
    - Fits a Gaussian to the prediction histogram
    - Saves histogram plot with Gaussian fit
4. At the end, plots:
    (a) μ - E_true vs E_true  (bias)
    (b) σ vs E_true           (resolution)
"""

import os
import json
import torch
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================================================================
# Configuration
# ================================================================
DATA_MODE = "mode4"
TARGET_MODE_NAME = "log_scaled_ratio"  # choose: direct_energy, scaled_ratio, log_scaled_ratio

ENERGY_VALUES = [20, 30, 40, 50, 100, 150, 200, 250, 300]
TOLERANCE = 0.5

# ================================================================
# Helper: Gaussian Function
# ================================================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# ================================================================
# Dataset Loader
# ================================================================
def load_dataset(data_mode):
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    dataset_map = {
        "mode4": "processed_data_large_v4.pt",
    }
    data_path = os.path.join(data_dir, dataset_map[data_mode])
    d = torch.load(data_path, map_location="cpu", weights_only=True)
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().reshape(-1).astype(np.float32)
    return X, y

# ================================================================
# Load Model
# ================================================================
def load_model():
    model_path = os.path.join(
        "models",
        f"model_{DATA_MODE}_{TARGET_MODE_NAME}",
        f"xgb_model_{DATA_MODE}_{TARGET_MODE_NAME}.json"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    print(f"[INFO] Loaded model from {model_path}")
    return bst

# ================================================================
# Gaussian Fit + Plot
# ================================================================
def plot_and_fit_histogram(pred_energies, E_true, tolerance, output_dir):
    bins = 40
    counts, bin_edges = np.histogram(pred_energies, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initial guess: amplitude, mean, std
    A0 = np.max(counts)
    mu0 = np.mean(pred_energies)
    sigma0 = np.std(pred_energies)

    try:
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[A0, mu0, sigma0])
        A_fit, mu_fit, sigma_fit = popt
    except RuntimeError:
        mu_fit, sigma_fit = np.nan, np.nan
        print(f"[WARN] Gaussian fit failed for E_true = {E_true}")
        return mu_fit, sigma_fit

    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(pred_energies, bins=bins, color="skyblue", alpha=0.7, edgecolor="black", label="Predictions")
    x_fit = np.linspace(min(pred_energies), max(pred_energies), 400)
    plt.plot(x_fit, gaussian(x_fit, *popt), "r-", label=f"Fit: μ={mu_fit:.2f}, σ={sigma_fit:.2f}")
    plt.axvline(E_true, color="k", linestyle="--", label="True Energy")
    plt.xlabel("Predicted Energy (GeV)")
    plt.ylabel("Counts")
    plt.title(f"Gaussian Fit — E_true={E_true} ± {tolerance}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"hist_fit_E{int(E_true)}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved histogram fit plot: {fig_path}")

    return mu_fit, sigma_fit

# ================================================================
# Main Validation Loop
# ================================================================
def validate():
    X, y = load_dataset(DATA_MODE)
    bst = load_model()

    output_dir = f"figures/validation_scan_{DATA_MODE}_{TARGET_MODE_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    mu_list, sigma_list, E_list = [], [], []

    for E_true in ENERGY_VALUES:
        mask = (y > E_true - TOLERANCE) & (y < E_true + TOLERANCE)
        X_sel, y_sel = X[mask], y[mask]
        if len(X_sel) == 0:
            print(f"[WARN] No samples near E_true = {E_true}")
            continue

        dsel = xgb.DMatrix(X_sel)
        preds = bst.predict(dsel)

        # ================================================================
        # Convert prediction to physical energy
        # ================================================================
        if TARGET_MODE_NAME == "direct_energy":
            E_pred = preds

        elif TARGET_MODE_NAME == "scaled_ratio":
            # E_pred = (y_pred / 100) × Σ first 28 fractional energies
            sum_frac = np.sum(X_sel[:, :28], axis=1)
            E_pred = (preds / 100.0) * sum_frac

        elif TARGET_MODE_NAME == "log_scaled_ratio":
            # E_pred = [(exp(y_pred) - 1) / 100] × Σ first 28 fractional energies
            sum_frac = np.sum(X_sel[:, :28], axis=1)
            E_pred = ((np.exp(preds) - 1.0) / 100.0) * sum_frac

        else:
            raise ValueError(f"Unknown TARGET_MODE_NAME '{TARGET_MODE_NAME}'")

        # ================================================================
        # Gaussian Fit
        # ================================================================
        mu_fit, sigma_fit = plot_and_fit_histogram(E_pred, E_true, TOLERANCE, output_dir)
        if not np.isnan(mu_fit):
            mu_list.append(mu_fit)
            sigma_list.append(sigma_fit)
            E_list.append(E_true)
            print(f"[INFO] E_true={E_true:6.1f}, μ_fit={mu_fit:8.3f}, σ_fit={sigma_fit:8.3f}, Δμ={mu_fit - E_true:8.3f}")

    # ================================================================
    # Summary Plots
    # ================================================================
    if len(E_list) == 0:
        print("[ERROR] No valid Gaussian fits found.")
        return

    # (1) μ - E_true vs E_true
    plt.figure(figsize=(8, 5))
    plt.plot(E_list, np.array(mu_list) - np.array(E_list), "o-", label="μ - E_true")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("μ_fit − E_true (GeV)")
    plt.title(f"Bias in Prediction Mean — {DATA_MODE} / {TARGET_MODE_NAME}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    bias_plot = os.path.join(output_dir, "mu_minus_Etrue_vs_E.png")
    plt.savefig(bias_plot, dpi=150)
    plt.close()
    print(f"[INFO] Saved bias plot: {bias_plot}")

    # (2) σ vs E_true
    plt.figure(figsize=(8, 5))
    plt.plot(E_list, sigma_list, "s-", color="purple", label="σ_fit")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("σ_fit (GeV)")
    plt.title(f"Energy Resolution — {DATA_MODE} / {TARGET_MODE_NAME}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    sigma_plot = os.path.join(output_dir, "sigma_vs_E.png")
    plt.savefig(sigma_plot, dpi=150)
    plt.close()
    print(f"[INFO] Saved sigma plot: {sigma_plot}")

    # ================================================================
    # Save Numerical Results
    # ================================================================
    results = {
        "mode": DATA_MODE,
        "target_mode": TARGET_MODE_NAME,
        "energies": E_list,
        "mu_fit": mu_list,
        "sigma_fit": sigma_list,
    }
    json_path = os.path.join(output_dir, "gaussian_fit_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved Gaussian fit summary: {json_path}")

# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    validate()
