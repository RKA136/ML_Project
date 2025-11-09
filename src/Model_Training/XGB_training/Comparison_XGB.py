#!/usr/bin/env python3
"""
Comparison_XGB.py
-----------------------------------
This script performs a quantitative comparison among three trained XGBoost models
that predict calorimeter energy reconstruction targets using different strategies.

Model descriptions:
 - Model_2: Directly predicts the reconstructed energy (E_pred_1).
 - Model_3: Predicts a multiplicative coefficient (E_pred_2 = coef * E_true).
 - Model_4: Predicts the natural logarithm of (1 + coefficient),
             then reconstructs E_pred_3 = (exp(ln_coef) - 1) * E_true.

Key steps:
1. Load preprocessed features (X) and true energies (y_true) from a .pt dataset file.
2. Load the three XGBoost models from the specified model directory.
3. Generate predictions for each model using a common DMatrix representation.
4. Compute reconstructed energies (E_pred_1, E_pred_2, E_pred_3) and their absolute errors.
5. Evaluate and print mean squared error (MSE) for each model.
6. Generate a Pandas DataFrame summarizing E_true, predictions, and errors.
7. Save a detailed comparison table to `model/model_comparison_with_errors.csv`.

This facilitates direct benchmarking between regression-based, coefficient-based,
and logarithmic-coefficient-based XGBoost approaches for energy prediction accuracy.
"""

import os
import json
import torch
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error

# ================================================================
# Configuration
# ================================================================
MODEL_DIR = "model"
NUM_SAMPLES = 100000   # number of samples to test
USE_GPU = False

model_path_1 = os.path.join(MODEL_DIR, "xgb_model_2.json")  # first model
model_path_2 = os.path.join(MODEL_DIR, "xgb_model_3.json")  # second model
model_path_3 = os.path.join(MODEL_DIR, "xgb_model_4.json")  # third model

# ================================================================
# Helper Functions
# ================================================================
def load_processed_tensors():
    """Load processed features and true energies."""
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    path = os.path.join(data_dir, "processed_data_large_v3.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")
    
    d = torch.load(path, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)
    y_true = d["y"].numpy().reshape(-1).astype(np.float32)
    return X, y_true


def load_model(path):
    """Load a trained XGBoost model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


# ================================================================
# Main Comparison
# ================================================================
def compare_models():
    # Load data
    X, y_true = load_processed_tensors()

    # Select first NUM_SAMPLES for comparison
    X_sel = X[:NUM_SAMPLES]
    y_sel = y_true[:NUM_SAMPLES]

    # Load models
    print("Loading trained models...")
    bst1 = load_model(model_path_1)
    bst2 = load_model(model_path_2)
    bst3 = load_model(model_path_3)

    # Convert to DMatrix
    dX = xgb.DMatrix(X_sel)

    # Generate predictions
    print("Generating predictions...")
    preds_1 = bst1.predict(dX)             # direct energy model
    coef_pred_2 = bst2.predict(dX)         # coefficient (%)
    ln_coef_pred_3 = bst3.predict(dX)      # ln of coefficient
    # Reconstruct energies
    E_pred_1 = preds_1
    E_pred_2 = (coef_pred_2) * y_sel
    E_pred_3 = (np.exp(ln_coef_pred_3)-1)*y_sel

    # Compute errors
    error_1 = E_pred_1 - y_sel
    error_2 = E_pred_2 - y_sel
    error_3 = E_pred_3 - y_sel
    abs_error_1 = np.abs(error_1)
    abs_error_2 = np.abs(error_2)
    abs_error_3 = np.abs(error_3)

    # Compute global MSEs
    mse_1 = mean_squared_error(y_sel, E_pred_1)
    mse_2 = mean_squared_error(y_sel, E_pred_2)
    mse_3 = mean_squared_error(y_sel, E_pred_3)

    # Tabulate results
    df = pd.DataFrame({
        "E_true": y_sel,
        "E_pred_1": E_pred_1,
        "Error_1": error_1,
        "|Error_1|": abs_error_1,
        "E_pred_2": E_pred_2,
        "Error_2": error_2,
        "|Error_2|": abs_error_2,
        "E_pred_3": E_pred_3,
        "Error_3": error_3,
        "|Error_3|": abs_error_3,
    })

    # Display
    print("\n=== Model Comparison (First {} Samples) ===".format(NUM_SAMPLES))
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    print("\nMSE (Model_2, E_pred_1): {:.6e}".format(mse_1))
    print("MSE (Model_3, E_pred_2): {:.6e}".format(mse_2))
    print("MSE (Model_4, E_pred_3): {:.6e}".format(mse_3))

    # Save results to CSV
    out_path = os.path.join(MODEL_DIR, "model_comparison_with_errors.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed comparison to: {out_path}")


# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    compare_models()
