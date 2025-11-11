#!/usr/bin/env python3
"""
XGB_train_multimode_final_v3.py
---------------------------------------------------
Unified XGBoost regression framework for calorimeter datasets.

Supports multiple preprocessing modes:
    mode1 → GPU-based (fractional + global)
    mode2 → CPU cumulant features
    mode3 → per-layer energy sums
    mode4 → layerwise [E_sum, E1/E7, E7/E19]

Core Capabilities:
------------------
1. Loads dataset tensors from `.pt` files (PyTorch format).
2. Splits data into Train / Validation / Test sets.
3. Provides 3 target definition options:
       (1) Direct prediction of true energy
       (2) 100 × (E_true / Σ first 28 fractional energies)
       (3) log(100 × (E_true / Σ first 28 fractional energies) + 1)
4. Trains an XGBoost regressor with early stopping.
5. Generates metrics, learning curves, and feature-importance plots.

Outputs:
---------
- model/xgb_model_<mode>.json
- figures_multimode/<mode>_metrics_vs_epochs.png
- figures_multimode/<mode>_feature_importance.png
- model/summary_<mode>.json
"""

import os
import json
import torch
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================================================================
# Configuration
# ================================================================
DATA_MODE = "mode4"          # choose among: mode1, mode2, mode3, mode4
MANUAL_DATA_PATH = None      # optional override for custom file

# Split ratios
TEST_SIZE = 0.10
VAL_SIZE = 0.10

NUM_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
USE_GPU = True
MODEL_DIR = "model"
FIGURES_DIR = "figures_multimode"
VERBOSE_EVAL = 25

# ================================================================
# Dataset Map
# ================================================================
def load_dataset(data_mode):
    # -------------------------------------------------------------------
    # Load configuration from config.json
    # -------------------------------------------------------------------
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    figures_dir = config.get("figures_dir", "figures")
    models_dir = config.get("models_dir", "model")

    # Update global paths dynamically
    global MODEL_DIR, FIGURES_DIR
    MODEL_DIR = models_dir
    FIGURES_DIR = figures_dir
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # Define mode → dataset mapping
    # -------------------------------------------------------------------
    dataset_map = {
        "mode1": {
            "path": "processed_data_large_v1.pt",
            "description": "v1: [E_layer_frac (28), E_sum, E_max, r_std, z_std]"
        },
        "mode2": {
            "path": "processed_data_large_v2.pt",
            "description": "v2: [E_layer_frac (28), r_cog–r_k5, z_cog–z_k5]"
        },
        "mode3": {
            "path": "processed_data_0001_v3.pt",
            "description": "v3: [E_layer_sum (28)] — per-layer energy sums"
        },
        "mode4": {
            "path": "processed_data_large_v4.pt",
            "description": "v4: [E_sum (28), E1/E7 (28), E7/E19 (28)]"
        },
    }

    if data_mode not in dataset_map:
        raise ValueError(f"Unknown DATA_MODE '{data_mode}'. Available: {list(dataset_map.keys())}")

    cfg = dataset_map[data_mode]

    # Construct absolute dataset path
    if MANUAL_DATA_PATH:
        data_path = MANUAL_DATA_PATH
    else:
        data_path = os.path.join(data_dir, cfg["path"])

    # -------------------------------------------------------------------
    # Validate path
    # -------------------------------------------------------------------
    abs_path = os.path.abspath(data_path)
    print(f"[INFO] Loading dataset for {data_mode}")
    print(f"[INFO] Description: {cfg['description']}")
    print(f"[INFO] Expected path: {abs_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {abs_path}")

    # -------------------------------------------------------------------
    # Load tensors
    # -------------------------------------------------------------------
    d = torch.load(data_path, map_location="cpu", weights_only=True)
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().reshape(-1).astype(np.float32)
    print(f"[INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")

    return X, y, data_mode


# ================================================================
# Helper Functions
# ================================================================
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    eps = 1e-8
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    rel = (y_pred - y_true) / denom
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mre = np.mean(rel)
    mare = np.mean(np.abs(rel))
    return mse, rmse, mae, mre, mare


# ================================================================
# Training Function
# ================================================================
def train():
    ensure_dirs()

    X, y, tag = load_dataset(DATA_MODE)
    n_samples, n_features = X.shape
    print(f"[INFO] Loaded {n_samples} samples with {n_features} features.")

    # ================================================================
    # Target Definition (Choose ONE)
    # ================================================================
    # (1) Direct prediction
    y_target = y

    # (2) Scaled ratio
    # y_target = 100.0 * (y / (np.sum(X[:, :28], axis=1) + 1e-8))

    # (3) Log-scaled ratio
    # y_target = np.log(100.0 * (y / (np.sum(X[:, :28], axis=1) + 1e-8)) + 1.0)

    # ================================================================
    # Data Splitting (Train / Validation / Test)
    # ================================================================
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_target, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_fraction, random_state=RANDOM_STATE)

    print(f"[INFO] Dataset split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val  : {len(X_val)} samples")
    print(f"  Test : {len(X_test)} samples")

    # ================================================================
    # Prepare DMatrix
    # ================================================================
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # ================================================================
    # XGBoost Parameters
    # ================================================================
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_STATE,
        "verbosity": 1,
    }
    if USE_GPU:
        xgb_params["tree_method"] = "gpu_hist"

    watchlist = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}

    print(f"\n[INFO] Starting XGBoost training (mode: {tag})...")
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=VERBOSE_EVAL,
    )

    # ================================================================
    # Model Saving
    # ================================================================
    model_path = os.path.join(MODEL_DIR, f"xgb_model_{tag}.json")
    bst.save_model(model_path)
    print(f"[INFO] Model saved to {model_path}")

    # ================================================================
    # Evaluation on Test Set
    # ================================================================
    preds = bst.predict(dtest)
    mse, rmse, mae, mre, mare = compute_metrics(y_test, preds)

    print("\n=== Test Metrics ===")
    print(f"MSE   : {mse:.6e}")
    print(f"RMSE  : {rmse:.6e}")
    print(f"MAE   : {mae:.6e}")
    print(f"MRE   : {mre:.6e}")
    print(f"MARE  : {mare:.6e}")

    # ================================================================
    # Plot Learning Curves
    # ================================================================
    epochs = np.arange(1, len(evals_result["train"]["rmse"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, evals_result["train"]["rmse"], label="Train RMSE")
    plt.plot(epochs, evals_result["validation"]["rmse"], label="Validation RMSE")
    plt.plot(epochs, evals_result["train"]["mae"], "--", label="Train MAE")
    plt.plot(epochs, evals_result["validation"]["mae"], "--", label="Validation MAE")
    plt.xlabel("Boosting Round")
    plt.ylabel("Error")
    plt.legend()
    plt.title(f"Training Curves ({tag})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{tag}_metrics_vs_epochs.png"), dpi=150)
    plt.close()

    # ================================================================
    # Feature Importance (all importance types)
    # ================================================================
    importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    importance_summary = {}

    for imp_type in importance_types:
        fmap = bst.get_score(importance_type=imp_type)
        importances = np.zeros(n_features, dtype=float)

        # Collect importance scores
        for k, v in fmap.items():
            if k.startswith("f"):
                idx = int(k[1:])
                if idx < n_features:
                    importances[idx] = v

        # Normalize if not empty
        if np.sum(importances) > 0:
            importances /= np.sum(importances)

        # Store numeric values for later summary
        importance_summary[imp_type] = importances.tolist()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(n_features), importances)
        plt.xticks(range(n_features), [f"f{i}" for i in range(n_features)], rotation=90)
        plt.xlabel("Feature Index")
        plt.ylabel("Normalized Importance")
        plt.title(f"Feature Importance ({imp_type}) — {tag}")
        plt.tight_layout()

        fi_path = os.path.join(FIGURES_DIR, f"{tag}_feature_importance_{imp_type}.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()

        print(f"[INFO] Saved {imp_type} feature importance plot: {fi_path}")

    # Save feature importance summary
    importance_json_path = os.path.join(MODEL_DIR, f"feature_importance_summary_{tag}.json")
    with open(importance_json_path, "w") as f:
        json.dump(importance_summary, f, indent=2)
    print(f"[INFO] Saved all importance values to {importance_json_path}")

    # ================================================================
    # Save Summary
    # ================================================================
    summary = {
        "mode": tag,
        "samples": n_samples,
        "features": n_features,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mre": mre,
        "mare": mare,
        "model_path": model_path,
        "feature_importance": fi_path,
    }
    with open(os.path.join(MODEL_DIR, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Training complete for {tag}. Summary saved.")


# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    train()
