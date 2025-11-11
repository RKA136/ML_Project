#!/usr/bin/env python3
"""
Compare_XGB_AllModes.py
---------------------------------------------------
Compares XGBoost regression models across multiple preprocessing modes and
target definitions for calorimeter energy prediction.

Each model pair corresponds to a specific combination of:
  - Data mode (mode1, mode2, mode3, mode4)
  - Target definition (direct_energy, scaled_ratio, log_scaled_ratio)

For each (mode, target):
  1. Loads the corresponding processed dataset (.pt file)
  2. Loads the trained XGBoost model from model/model_<mode>_<target>/
  3. Generates predictions
  4. If target is coefficient-based, reconstructs true energy
  5. Computes metrics (MSE, RMSE, MAE)
  6. Saves a detailed comparison CSV and summary JSON

Output folders follow the structure:
  models/model_<mode>_<target>/
  figures/figures_<mode>_<target>/
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================================================================
# CONFIGURATION
# ================================================================
NUM_SAMPLES = 50000
USE_GPU = False

# Model/Mode combinations
DATA_MODES = ["mode1", "mode2", "mode3", "mode4"]
TARGET_MODES = ["direct_energy", "scaled_ratio", "log_scaled_ratio"]

# ================================================================
# UTILITIES
# ================================================================
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def load_data_for_mode(data_mode, data_dir):
    """Load dataset corresponding to mode."""
    dataset_map = {
        "mode1": "processed_data_large_v1.pt",
        "mode2": "processed_data_large_v2.pt",
        "mode3": "processed_data_0001_v3.pt",
        "mode4": "processed_data_large_v4.pt",
    }

    if data_mode not in dataset_map:
        raise ValueError(f"Unknown mode: {data_mode}")

    data_path = os.path.join(data_dir, dataset_map[data_mode])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    d = torch.load(data_path, map_location="cpu", weights_only=True)
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().reshape(-1).astype(np.float32)
    return X, y

def load_model_for_mode(model_dir, mode, target):
    model_path = os.path.join(model_dir, f"model_{mode}_{target}", f"xgb_model_{mode}_{target}.json")
    if not os.path.exists(model_path):
        print(f"[WARNING] Model missing: {model_path}")
        return None
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

# ================================================================
# MAIN COMPARISON
# ================================================================
def compare_all_models():
    config = load_config()
    data_dir = config.get("data_dir", ".")
    model_root = config.get("models_dir", "model")

    results_summary = []

    for mode in DATA_MODES:
        print(f"\n============================")
        print(f" Comparing Models for {mode}")
        print(f"============================")

        # Load dataset
        try:
            X, y_true = load_data_for_mode(mode, data_dir)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

        X = X[:NUM_SAMPLES]
        y_true = y_true[:NUM_SAMPLES]
        dX = xgb.DMatrix(X)

        # Collect predictions for all target types
        preds_dict = {}
        for target in TARGET_MODES:
            bst = load_model_for_mode(model_root, mode, target)
            if bst is None:
                continue

            y_pred_raw = bst.predict(dX)

            # Reconstruction logic depends on target type
            if target == "direct_energy":
                y_pred = y_pred_raw
            elif target == "scaled_ratio":
                y_pred = (y_pred_raw) * y_true
            elif target == "log_scaled_ratio":
                y_pred = (np.exp(y_pred_raw) - 1) * y_true
            else:
                raise ValueError(f"Unknown target mode: {target}")

            preds_dict[target] = y_pred

        # Skip modes with missing models
        if not preds_dict:
            print(f"[INFO] No trained models found for {mode}")
            continue

        # Build comparison table
        df = pd.DataFrame({"E_true": y_true})
        for target, preds in preds_dict.items():
            df[f"E_pred_{target}"] = preds
            df[f"Error_{target}"] = preds - y_true
            df[f"|Error_{target}|"] = np.abs(preds - y_true)

        # Compute metrics
        metrics = {}
        for target, preds in preds_dict.items():
            mse, rmse, mae = compute_metrics(y_true, preds)
            metrics[target] = {"MSE": mse, "RMSE": rmse, "MAE": mae}

        # Save results
        out_dir = os.path.join(model_root, f"model_{mode}_comparisons")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"comparison_{mode}.csv")
        df.to_csv(csv_path, index=False)

        json_path = os.path.join(out_dir, f"summary_{mode}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[SAVED] Comparison CSV: {csv_path}")
        print(f"[SAVED] Summary JSON : {json_path}")

        # Print summary to console
        print("\nPerformance Summary:")
        for target, vals in metrics.items():
            print(f"  {target:<18s}  MSE={vals['MSE']:.4e}  RMSE={vals['RMSE']:.4e}  MAE={vals['MAE']:.4e}")

        results_summary.append({"mode": mode, "metrics": metrics})

    # Save global summary
    global_summary = os.path.join(model_root, "comparison_global_summary.json")
    with open(global_summary, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[INFO] Global summary saved: {global_summary}")

# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    compare_all_models()
