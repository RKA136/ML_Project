#!/usr/bin/env python3
"""
XGB_residual_analysis.py
-----------------------------------
Performs residual diagnostics and feature interaction visualization
for the trained XGBoost regression model.

Inputs:
 - model/xgb_model.json
 - config.json → to locate processed_data_large_v3.pt

Outputs:
 - figures/residual_distribution.png
 - figures/residuals_vs_predicted.png
 - figures/predicted_vs_true.png
 - figures/residuals_vs_fXX.png (for top features)
 - figures/interaction_fA_fB.png (optional)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ================================================================
# Configuration
# ================================================================
TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42
MODEL_PATH = "model/xgb_model.json"
FIGURES_DIR = "figures"

# ================================================================
# Helper Functions
# ================================================================
def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def load_processed_tensors():
    """Load processed tensors using config.json"""
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    path = os.path.join(data_dir, "processed_data_large_v3.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")
    d = torch.load(path, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().reshape(-1).astype(np.float32)
    return X, y

def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

# ================================================================
# Residual Analysis
# ================================================================
def residual_analysis():
    ensure_dirs()
    print("Loading model and data...")
    bst = load_model()
    X, y = load_processed_tensors()
    n_samples, n_features = X.shape

    # Use same split logic as training
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_frac, random_state=RANDOM_STATE)

    print(f"Data partitions -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    dtest = xgb.DMatrix(X_test)
    y_pred = bst.predict(dtest)
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)

    # ================================================================
    # Numerical Diagnostics
    # ================================================================
    r2 = r2_score(y_test, y_pred)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    mean_abs_res = np.mean(abs_residuals)
    skew_res = np.mean(((residuals - mean_res) / std_res) ** 3)
    kurt_res = np.mean(((residuals - mean_res) / std_res) ** 4) - 3

    print("\n=== Residual Statistics ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Residual: {mean_res:.4e}")
    print(f"Std Dev of Residuals: {std_res:.4e}")
    print(f"Mean |Residual|: {mean_abs_res:.4e}")
    print(f"Skewness: {skew_res:.4f}")
    print(f"Kurtosis: {kurt_res:.4f}")

    # ================================================================
    # Plots
    # ================================================================
    # 1. Residual Distribution
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=80, color="steelblue", alpha=0.7, edgecolor="black")
    plt.title("Residual Distribution", fontsize=14)
    plt.xlabel("Residual (y_true − y_pred)")
    plt.ylabel("Frequency")
    plt.axvline(0, color="red", linestyle="--", lw=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residual_distribution.png"), dpi=200)
    plt.close()

    # 2. Residuals vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, s=8, alpha=0.4, color="teal")
    plt.axhline(0, color="red", linestyle="--", lw=1)
    plt.title("Residuals vs Predicted", fontsize=14)
    plt.xlabel("Predicted Value (ŷ)")
    plt.ylabel("Residual (y − ŷ)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_vs_predicted.png"), dpi=200)
    plt.close()

    # 3. Predicted vs True
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=8, alpha=0.4, color="darkorange")
    lims = [min(y_test), max(y_test)]
    plt.plot(lims, lims, "r--", lw=2)
    plt.title("Predicted vs True", fontsize=14)
    plt.xlabel("True Value (y)")
    plt.ylabel("Predicted Value (ŷ)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "predicted_vs_true.png"), dpi=200)
    plt.close()

    # ================================================================
    # Residuals vs Top Features
    # ================================================================
    fmap = bst.get_score(importance_type="weight")
    importances = np.zeros(n_features)
    for k, v in fmap.items():
        if k.startswith("f"):
            idx = int(k[1:])
            if idx < n_features:
                importances[idx] = v
    if importances.sum() > 0:
        importances /= importances.sum()

    top_indices = np.argsort(importances)[::-1][:3]
    print(f"\nTop features by importance: {[f'f{i}' for i in top_indices]}")

    for idx in top_indices:
        plt.figure(figsize=(7, 5))
        plt.scatter(X_test[:, idx], residuals, s=8, alpha=0.4, color="navy")
        plt.axhline(0, color="red", linestyle="--", lw=1)
        plt.title(f"Residuals vs f{idx}", fontsize=14)
        plt.xlabel(f"Feature f{idx}")
        plt.ylabel("Residual (y − ŷ)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"residuals_vs_f{idx}.png"), dpi=200)
        plt.close()

    # ================================================================
    # Feature Interaction (2D KDE)
    # ================================================================
    if len(top_indices) >= 2:
        i1, i2 = top_indices[:2]
        plt.figure(figsize=(7, 6))
        sns.kdeplot(
            x=X_test[:, i1],
            y=X_test[:, i2],
            fill=True,
            cmap="viridis",
            levels=100,
            weights=y_pred,
            alpha=0.8,
        )
        plt.title(f"Feature Interaction: f{i1} vs f{i2}", fontsize=14)
        plt.xlabel(f"f{i1}")
        plt.ylabel(f"f{i2}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"interaction_f{i1}_f{i2}.png"), dpi=200)
        plt.close()

    print("\nResidual analysis complete. Plots saved in 'figures/' directory.")

# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    residual_analysis()
