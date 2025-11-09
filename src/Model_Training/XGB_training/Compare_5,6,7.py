#!/usr/bin/env python3
"""
Comparison_XGB_v7.py
-----------------------------------
This script performs a quantitative comparison among three trained XGBoost models:
    - v5: Direct energy regression
    - v6: Ratio-based regression (E_true / Σ(layer energies))
    - v7: Logarithmic ratio regression (log(1 + 100×E_true/Σ(layer energies)))

All models are evaluated on the same processed dataset and compared using
reconstructed energy predictions (E_pred_5, E_pred_6, E_pred_7).

Outputs:
---------
- Prints mean squared error (MSE) and mean absolute error (MAE)
  for each version.
- Generates a detailed comparison DataFrame.
- Saves results to `model/model_comparison_v5_v7.csv`.

Author: Adapted from Comparison_XGB.py
"""

import os
import json
import torch
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================================================================
# Configuration
# ================================================================
MODEL_DIR = "model"
NUM_SAMPLES = 100000  # number of samples to evaluate
USE_GPU = False

model_path_v5 = os.path.join(MODEL_DIR, "xgb_model_5.json")
model_path_v6 = os.path.join(MODEL_DIR, "xgb_model_6.json")
model_path_v7 = os.path.join(MODEL_DIR, "xgb_model_7.json")

# ================================================================
# Helper Functions
# ================================================================
def load_processed_tensors():
    """Load processed calorimeter features and true energies."""
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    path = os.path.join(data_dir, "processed_data_0001_v4.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")

    d = torch.load(path, map_location="cpu", weights_only=True)
    X = d["X"].numpy().astype(np.float32)
    y_true = d["y"].numpy().reshape(-1).astype(np.float32)
    return X, y_true


def load_model(path):
    """Load a trained XGBoost booster model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


# ================================================================
# Main Comparison
# ================================================================
def compare_v5_v6_v7():
    print("Loading processed data...")
    X, y_true = load_processed_tensors()

    # Select subset
    X_sel = X[:NUM_SAMPLES]
    y_sel = y_true[:NUM_SAMPLES]
    sum_frac_layers = np.sum(X_sel[:, :28], axis=1) + 1e-8  # sum over first 28 features

    # Load models
    print("Loading trained models (v5, v6, v7)...")
    bst5 = load_model(model_path_v5)
    bst6 = load_model(model_path_v6)
    bst7 = load_model(model_path_v7)

    dX = xgb.DMatrix(X_sel)

    # Generate predictions
    print("Generating predictions...")
    preds_v5 = bst5.predict(dX)      # Direct energy regression
    preds_v6 = bst6.predict(dX)      # Ratio-based
    preds_v7 = bst7.predict(dX)      # Log-ratio-based

    # Reconstruct energies
    E_pred_5 = preds_v5
    E_pred_6 = (preds_v6 / 100.0) * sum_frac_layers
    E_pred_7 = ((np.exp(preds_v7) - 1.0) / 100.0) * sum_frac_layers

    # Compute errors
    err_5 = E_pred_5 - y_sel
    err_6 = E_pred_6 - y_sel
    err_7 = E_pred_7 - y_sel

    abs_err_5 = np.abs(err_5)
    abs_err_6 = np.abs(err_6)
    abs_err_7 = np.abs(err_7)

    # Compute metrics
    mse_5 = mean_squared_error(y_sel, E_pred_5)
    mse_6 = mean_squared_error(y_sel, E_pred_6)
    mse_7 = mean_squared_error(y_sel, E_pred_7)

    mae_5 = mean_absolute_error(y_sel, E_pred_5)
    mae_6 = mean_absolute_error(y_sel, E_pred_6)
    mae_7 = mean_absolute_error(y_sel, E_pred_7)

    # Tabulate results
    df = pd.DataFrame({
        "E_true": y_sel,
        "E_pred_v5": E_pred_5,
        "Error_v5": err_5,
        "|Error_v5|": abs_err_5,
        "E_pred_v6": E_pred_6,
        "Error_v6": err_6,
        "|Error_v6|": abs_err_6,
        "E_pred_v7": E_pred_7,
        "Error_v7": err_7,
        "|Error_v7|": abs_err_7,
    })

    # Print summary
    print("\n=== Model Comparison (First {} Samples) ===".format(NUM_SAMPLES))
    print(f"MSE_v5 (Direct energy)         : {mse_5:.6e}")
    print(f"MSE_v6 (Scaled ratio)          : {mse_6:.6e}")
    print(f"MSE_v7 (Logarithmic ratio)     : {mse_7:.6e}")
    print(f"MAE_v5                         : {mae_5:.6e}")
    print(f"MAE_v6                         : {mae_6:.6e}")
    print(f"MAE_v7                         : {mae_7:.6e}")

    # Save DataFrame
    out_path = os.path.join(MODEL_DIR, "model_comparison_v5_v7.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed comparison to: {out_path}")

    # Print top 10 rows
    print("\n--- Preview (Top 10) ---")
    print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # Summary metrics dataframe
    summary = pd.DataFrame({
        "Model": ["v5", "v6", "v7"],
        "MSE": [mse_5, mse_6, mse_7],
        "MAE": [mae_5, mae_6, mae_7]
    })
    print("\n=== Summary Table ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:10.6e}"))

    summary_path = os.path.join(MODEL_DIR, "model_comparison_summary_v5_v7.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to: {summary_path}")


# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    compare_v5_v6_v7()
