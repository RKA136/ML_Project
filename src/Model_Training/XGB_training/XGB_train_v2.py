#!/usr/bin/env python3
"""
XGB_train_v2.py
-----------------------------------
Train an **XGBoost regressor** on calorimeter feature data stored in `processed_data.pt`,
evaluate model performance using multiple error metrics, and visualize training evolution.

Pipeline Overview:
------------------
1. **Configuration and Setup**
   - Flexible configuration through constants and `config.json`.
   - Supports GPU acceleration, early stopping, and optional feature scaling.

2. **Data Loading**
   - Loads preprocessed calorimeter event data from `processed_data_large_v3.pt`.
   - Data format: `{"X": features, "y": true_energies}` stored as PyTorch tensors.

3. **Dataset Partitioning**
   - Splits data into training, validation, and test subsets:
        80% training
        10% validation
        10% testing
   - Ensures reproducibility with fixed random seed.

4. **XGBoost Training**
   - Uses regression objective (`reg:squarederror`) with RMSE and MAE metrics.
   - Tracks performance on training and validation sets across boosting rounds.
   - Early stopping halts training when validation performance stagnates.

5. **Performance Metrics**
   - MSE: Mean Squared Error
   - RMSE: Root Mean Squared Error
   - MAE: Mean Absolute Error
   - MRE: Mean Relative Error
   - MARE: Mean Absolute Relative Error

6. **Visualization Outputs**
   - `metrics_vs_epochs.png` → RMSE and MAE vs epoch
   - `relative_error_vs_epochs.png` → validation MRE and MARE over epochs
   - `feature_importance.png` → normalized feature importance (layer-wise)

7. **Model and Output Artifacts**
   - Saves trained model as `model/xgb_model_2.json`
   - Saves diagnostic plots in `figures/`
   - Prints key performance indicators and ranked feature importances

Use Case:
---------
Optimized for calorimeter energy reconstruction studies where layer-wise and
aggregate features are used for regression-based prediction of true shower energy.

Example Run:
------------
    python XGB_train_v2.py

Output:
--------
- Model file: `model/xgb_model_2.json`
- Plots: `figures/metrics_vs_epochs.png`, `figures/relative_error_vs_epochs.png`, `figures/feature_importance.png`
- Console metrics summary
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
MODEL_DIR = "model"
FIGURES_DIR = "figures"
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
    """Custom evaluation metric for XGBoost."""
    labels = dtrain.get_label()
    eps = 1e-8
    denom = np.where(np.abs(labels) < eps, eps, labels)
    mare = np.mean(np.abs((preds - labels) / denom))
    return "mare", float(mare)


def load_processed_tensors():
    """Load processed features and labels from processed_data.pt"""
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


# ================================================================
# Training and Evaluation
# ================================================================
def train():
    ensure_dirs()
    print("Loading processed data...")
    X, y = load_processed_tensors()
    n_samples, n_features = X.shape
    print(f"Loaded {n_samples} samples with {n_features} features.")

    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_frac, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Optional scaling
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

    print("Starting XGBoost training...")
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=VERBOSE_EVAL,
    )

    # Save model
    model_path = os.path.join(MODEL_DIR, "xgb_model_2.json")
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Predictions on test data
    preds_test = bst.predict(dtest)
    eps = 1e-8
    denom = np.where(np.abs(y_test) < eps, eps, y_test)
    rel_errors = (preds_test - y_test) / denom

    # Compute metrics
    mse = mean_squared_error(y_test, preds_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds_test)
    mre = np.mean(rel_errors)
    mare = np.mean(np.abs(rel_errors))

    print("\n=== Test Set Metrics ===")
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

    # Recompute validation MRE & MARE for each epoch
    val_mre_list = []
    val_mare_list = []
    for r in range(1, rounds + 1):
        preds_r = bst.predict(dval, iteration_range=(0, r))
        denom = np.where(np.abs(y_val) < eps, eps, y_val)
        rel = (preds_r - y_val) / denom
        val_mre_list.append(np.mean(rel))
        val_mare_list.append(np.mean(np.abs(rel)))

    # ---- RMSE & MAE ----
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label="Train RMSE")
    plt.plot(epochs, val_rmse, label="Val RMSE")
    plt.plot(epochs, train_mae, "--", label="Train MAE")
    plt.plot(epochs, val_mae, "--", label="Val MAE")
    plt.xlabel("Epoch (Boosting Round)")
    plt.ylabel("Error")
    plt.title("Training and Validation Error vs Epoch")
    plt.legend()
    plt.grid(True)
    metrics_path = os.path.join(FIGURES_DIR, "metrics_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"Saved metrics plot: {metrics_path}")

    # ---- Relative Errors ----
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_mre_list, label="Val MRE (signed)")
    plt.plot(epochs, val_mare_list, "--", label="Val MARE (abs)")
    plt.xlabel("Epoch (Boosting Round)")
    plt.ylabel("Relative Error")
    plt.title("Validation Relative Errors vs Epoch")
    plt.legend()
    plt.grid(True)
    rel_path = os.path.join(FIGURES_DIR, "relative_error_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(rel_path, dpi=150)
    plt.close()
    print(f"Saved relative error plot: {rel_path}")

    # ================================================================
    # Feature Importance (Sequential, not sorted)
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
    plt.xlabel("Layer / Feature Index")
    plt.ylabel("Normalized Importance")
    plt.title("Feature Importance by Layer (Sequential Order)")
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
    print("\nTraining complete.")
