#!/usr/bin/env python3
"""
XGB_train_v6.py
-----------------------------------
Train an XGBoost regressor on **v6-processed calorimeter features**
to predict the scaled energy ratio target:

    target = 100 × (E_true / Σ(E_layer_fractional))

Here:
    - The first 28 columns of X correspond to fractional energy deposits
      across calorimeter layers.
    - E_true is the total true event energy (ground truth label).

Objective:
----------
v6 version isolates the summation of fractional energies to the first 28
features, providing a more physically meaningful target definition for
energy calibration.

Outputs:
---------
- figures_v6/metrics_vs_epochs.png
- figures_v6/relative_error_vs_epochs.png
- figures_v6/feature_importance.png
- model_v6/xgb_model_v6.json

Author: Adapted from XGB_train_v5.py
"""

import os
import json
import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ================================================================
# Configuration
# ================================================================
NUM_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20
TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42
USE_GPU = False
SCALE_FEATURES = False
MODEL_DIR = "model"           # [CHANGED in v6]
FIGURES_DIR = "figures_v6"       # [CHANGED in v6]
VERBOSE_EVAL = 10

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


# ================================================================
# Helper Functions
# ================================================================
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def mean_absolute_relative_error(preds, dtrain):
    labels = dtrain.get_label()
    eps = 1e-8
    denom = np.where(np.abs(labels) < eps, eps, labels)
    mare = np.mean(np.abs((preds - labels) / denom))
    return "mare", float(mare)


def load_processed_tensors():
    """
    Load processed features and labels from processed_data_0001_v4.pt (v6 input)

    [CHANGED in v6]
    The target is redefined as:
        y = 100 × (E_true / Σ(E_layer_fractional))
    where Σ(E_layer_fractional) = sum of the first 28 columns of X.
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    path = os.path.join(data_dir, "processed_data_0001_v4.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")

    d = torch.load(path, map_location="cpu", weights_only=True)
    X = d["X"].numpy().astype(np.float32)
    y_true = d["y"].numpy().reshape(-1).astype(np.float32)

    # [ADDED in v6] Sum only first 28 columns for fractional energy layers
    sum_frac_layers = np.sum(X[:, :28], axis=1) + 1e-8  # avoid zero division

    # [ADDED in v6] Define new target ratio
    y = 100.0 * (y_true / sum_frac_layers).astype(np.float32)

    return X, y


# ================================================================
# Training and Evaluation
# ================================================================
def train():
    ensure_dirs()
    print("Loading processed data (v6 ratio target)...")
    X, y = load_processed_tensors()
    n_samples, n_features = X.shape
    print(f"Loaded {n_samples} samples with {n_features} features.")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_frac, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler = None
    if SCALE_FEATURES:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}

    print("Starting XGBoost training (v6 ratio target)...")
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=VERBOSE_EVAL,
    )

    # [CHANGED in v6] Save path
    model_path = os.path.join(MODEL_DIR, "xgb_model_6.json")
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")

    preds_test = bst.predict(dtest)
    eps = 1e-8
    denom = np.where(np.abs(y_test) < eps, eps, y_test)
    rel_errors = (preds_test - y_test) / denom

    mse = mean_squared_error(y_test, preds_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds_test)
    mre = np.mean(rel_errors)
    mare = np.mean(np.abs(rel_errors))

    print("\n=== Test Set Metrics (v6 ratio target) ===")
    print(f"MSE   : {mse:.6e}")
    print(f"RMSE  : {rmse:.6e}")
    print(f"MAE   : {mae:.6e}")
    print(f"MRE   : {mre:.6e}")
    print(f"MARE  : {mare:.6e}")

    # ================================================================
    # Plot Metrics vs Epochs
    # ================================================================
    train_rmse = evals_result.get("train", {}).get("rmse", [])
    val_rmse = evals_result.get("validation", {}).get("rmse", [])
    train_mae = evals_result.get("train", {}).get("mae", [])
    val_mae = evals_result.get("validation", {}).get("mae", [])

    rounds = len(train_rmse)
    epochs = np.arange(1, rounds + 1)

    val_mre_list = []
    val_mare_list = []
    for r in range(1, rounds + 1):
        preds_r = bst.predict(dval, iteration_range=(0, r))
        denom = np.where(np.abs(y_val) < eps, eps, y_val)
        rel = (preds_r - y_val) / denom
        val_mre_list.append(np.mean(rel))
        val_mare_list.append(np.mean(np.abs(rel)))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label="Train RMSE")
    plt.plot(epochs, val_rmse, label="Val RMSE")
    plt.plot(epochs, train_mae, "--", label="Train MAE")
    plt.plot(epochs, val_mae, "--", label="Val MAE")
    plt.xlabel("Epoch (Boosting Round)")
    plt.ylabel("Error")
    plt.title("Training and Validation Error vs Epoch (v6 ratio target)")
    plt.legend()
    plt.grid(True)
    metrics_path = os.path.join(FIGURES_DIR, "metrics_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"Saved metrics plot: {metrics_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_mre_list, label="Val MRE (signed)")
    plt.plot(epochs, val_mare_list, "--", label="Val MARE (abs)")
    plt.xlabel("Epoch (Boosting Round)")
    plt.ylabel("Relative Error")
    plt.title("Validation Relative Errors vs Epoch (v6 ratio target)")
    plt.legend()
    plt.grid(True)
    rel_path = os.path.join(FIGURES_DIR, "relative_error_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(rel_path, dpi=150)
    plt.close()
    print(f"Saved relative error plot: {rel_path}")

    # ================================================================
    # Feature Importance
    # ================================================================
    fmap = bst.get_score(importance_type="weight")
    importances = np.zeros(n_features, dtype=float)
    for k, v in fmap.items():
        if k.startswith("f"):
            idx = int(k[1:])
            if idx < n_features:
                importances[idx] = v
    if importances.sum() > 0:
        importances /= importances.sum()

    plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), importances)
    plt.xticks(range(n_features), [f"f{i}" for i in range(n_features)], rotation=90)
    plt.xlabel("Feature Index (Sequential)")
    plt.ylabel("Normalized Importance")
    plt.title("Feature Importance (v6 ratio target)")
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot: {fi_path}")

    print("\nFeature Importances (Sequential):")
    for i in range(n_features):
        print(f" f{i}: {importances[i]:.4f}")

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mre": mre,
        "mare": mare,
        "plots": {
            "metrics": metrics_path,
            "relative": rel_path,
            "importance": fi_path
        }
    }
    return results


# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    results = train()
    print("\nTraining complete (v6 ratio target).")
