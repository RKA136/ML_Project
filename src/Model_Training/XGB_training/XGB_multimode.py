#!/usr/bin/env python3
"""
XGB_train_multimode.py
---------------------------------------------------
Unified XGBoost regression framework for calorimeter datasets.

Supports preprocessing modes:
    mode1 → GPU-based (fractional + global)
    mode2 → CPU cumulant features
    mode3 → per-layer energy sums
    mode4 → layerwise [E_sum, E1/E7, E7/E19]

Target modes:
    (1) Direct energy prediction
    (2) 100 × (E_true / Σ first 28 fractional energies)
    (3) log(100 × (E_true / Σ first 28 fractional energies) + 1)

Output folder structure:
    models/model_<mode>_<target>/
        ├── xgb_model_<mode>_<target>.json
        ├── summary_<mode>_<target>.json
        ├── feature_importance_summary_<mode>_<target>.json

    figures/figures_<mode>_<target>/
        ├── <mode>_<target>_metrics_vs_epochs.png
        ├── <mode>_<target>_feature_importance_*.png
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
MANUAL_DATA_PATH = None      # optional override

TEST_SIZE = 0.10
VAL_SIZE = 0.10
NUM_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
USE_GPU = True
VERBOSE_EVAL = 25


# ================================================================
# Dataset Loader
# ================================================================
def load_dataset(data_mode):
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    figures_dir = config.get("figures_dir", "figures")
    models_dir = config.get("models_dir", "models")

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
    data_path = MANUAL_DATA_PATH or os.path.join(data_dir, cfg["path"])
    abs_path = os.path.abspath(data_path)

    print(f"[INFO] Loading dataset for {data_mode}")
    print(f"[INFO] Description: {cfg['description']}")
    print(f"[INFO] Path: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Data file not found at: {abs_path}")

    d = torch.load(abs_path, map_location="cpu", weights_only=True)
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().reshape(-1).astype(np.float32)

    print(f"[INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, data_mode, figures_dir, models_dir


# ================================================================
# Helper Functions
# ================================================================
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
    X, y, tag, FIGURES_ROOT, MODELS_ROOT = load_dataset(DATA_MODE)
    n_samples, n_features = X.shape
    print(f"[INFO] Loaded {n_samples} samples with {n_features} features.")

    # ================================================================
    # Target Definition (choose ONE)
    # ================================================================
    # (1) Direct prediction
    # y_target = y
    # TARGET_MODE_NAME = "direct_energy"

    # (2) Scaled ratio
    # y_target = 100.0 * (y / (np.sum(X[:, :28], axis=1) + 1e-8))
    # TARGET_MODE_NAME = "scaled_ratio"

    # (3) Log-scaled ratio
    y_target = np.log(100.0 * (y / (np.sum(X[:, :28], axis=1) + 1e-8)) + 1.0)
    TARGET_MODE_NAME = "log_scaled_ratio"

    print(f"[INFO] Selected target mode: {TARGET_MODE_NAME}")

    # ================================================================
    # Directory Setup
    # ================================================================
    FIGURES_SUBDIR = os.path.join(FIGURES_ROOT, f"figures_{tag}_{TARGET_MODE_NAME}")
    MODELS_SUBDIR = os.path.join(MODELS_ROOT, f"model_{tag}_{TARGET_MODE_NAME}")
    os.makedirs(FIGURES_SUBDIR, exist_ok=True)
    os.makedirs(MODELS_SUBDIR, exist_ok=True)
    print(f"[INFO] Figures folder: {FIGURES_SUBDIR}")
    print(f"[INFO] Models folder : {MODELS_SUBDIR}")

    # ================================================================
    # Data Splitting
    # ================================================================
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_target, test_size=0.10, random_state=RANDOM_STATE)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_fraction, random_state=RANDOM_STATE)

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
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
        params["tree_method"] = "gpu_hist"

    watchlist = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}

    print(f"\n[INFO] Starting XGBoost training — {tag} ({TARGET_MODE_NAME})")
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=VERBOSE_EVAL,
    )

    # ================================================================
    # Model Save Paths
    # ================================================================
    model_path = os.path.join(MODELS_SUBDIR, f"xgb_model_{tag}_{TARGET_MODE_NAME}.json")
    summary_path = os.path.join(MODELS_SUBDIR, f"summary_{tag}_{TARGET_MODE_NAME}.json")
    importance_json_path = os.path.join(MODELS_SUBDIR, f"feature_importance_summary_{tag}_{TARGET_MODE_NAME}.json")

    bst.save_model(model_path)
    print(f"[INFO] Model saved to: {model_path}")

    # ================================================================
    # Evaluation
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
    # Learning Curve Plot
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
    plt.title(f"Training Curves — {tag} ({TARGET_MODE_NAME})")
    plt.grid(True)
    metrics_path = os.path.join(FIGURES_SUBDIR, f"{tag}_{TARGET_MODE_NAME}_metrics_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved metrics plot: {metrics_path}")

    # ================================================================
    # Feature Importances
    # ================================================================
    importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    importance_summary = {}

    for imp_type in importance_types:
        fmap = bst.get_score(importance_type=imp_type)
        importances = np.zeros(n_features, dtype=float)
        for k, v in fmap.items():
            if k.startswith("f"):
                idx = int(k[1:])
                if idx < n_features:
                    importances[idx] = v
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
        importance_summary[imp_type] = importances.tolist()

        plt.figure(figsize=(12, 6))
        plt.bar(range(n_features), importances)
        plt.xticks(range(n_features), [f"f{i}" for i in range(n_features)], rotation=90)
        plt.xlabel("Feature Index")
        plt.ylabel("Normalized Importance")
        plt.title(f"Feature Importance ({imp_type}) — {tag} ({TARGET_MODE_NAME})")
        plt.tight_layout()
        fi_path = os.path.join(FIGURES_SUBDIR, f"{tag}_{TARGET_MODE_NAME}_feature_importance_{imp_type}.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved feature importance plot ({imp_type}): {fi_path}")

    with open(importance_json_path, "w") as f:
        json.dump(importance_summary, f, indent=2)
    print(f"[INFO] Saved importance summary: {importance_json_path}")

    # ================================================================
    # Save Summary JSON
    # ================================================================
    summary = {
        "mode": tag,
        "target_mode": TARGET_MODE_NAME,
        "samples": int(n_samples),
        "features": int(n_features),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mre": float(mre),
        "mare": float(mare),
        "model_path": model_path,
        "metrics_plot": metrics_path,
        "feature_importance_json": importance_json_path,
        "figures_dir": FIGURES_SUBDIR,
        "models_dir": MODELS_SUBDIR,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved: {summary_path}")


# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    train()
