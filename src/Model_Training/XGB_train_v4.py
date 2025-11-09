#!/usr/bin/env python3
"""
XGB_train_v2.py
-----------------------------------
Train an XGBoost regressor to predict:
    y_target = 100 × (E_true / Σ(layer_energies))

The model learns a correction factor (in %) between the true energy
and the total recorded energy in all 28 layers.

It also reconstructs E_pred = (pred / 100) × Σ(layer_energies)
for physical comparison with E_true.

Evaluates with MSE, RMSE, MAE, and relative error metrics.

Generates:
 - figures/metrics_vs_epochs.png
 - figures/relative_error_vs_epochs.png
 - figures/feature_importance.png
 - model/xgb_model.json
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
    """
    Load processed features and redefine target as:
        y_target = 100 * (E_true / Σ(layer_energies))
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    path = os.path.join(data_dir, "processed_data_large_v3.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")
    
    d = torch.load(path, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)
    y_true = d["y"].numpy().reshape(-1).astype(np.float32)

    # --- compute sum of 28 layer energies for each event ---
    layer_sum = np.sum(X[:, :28], axis=1) + 1e-8 # adjust slice if needed
    # --- redefine target ---
    y = np.log(((y_true *100)/ layer_sum)+1)

    print(f"Redefined target as E_target = 100 × (E_true / Σ(layer_energies)), shape = {y.shape}")
    return X, y, y_true, layer_sum


# ================================================================
# Training and Evaluation
# ================================================================
def train():
    ensure_dirs()
    print("Loading processed data...")
    X, y, y_true, layer_sum = load_processed_tensors()
    n_samples, n_features = X.shape
    print(f"Loaded {n_samples} samples with {n_features} features.")

    # Split dataset
    X_temp, X_test, y_temp, y_test, ytrue_temp, ytrue_test, lsum_temp, lsum_test = train_test_split(
        X, y, y_true, layer_sum, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val, ytrue_train, ytrue_val, lsum_train, lsum_val = train_test_split(
        X_temp, y_temp, ytrue_temp, lsum_temp, test_size=val_frac, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Optional scaling
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
    model_path = os.path.join(MODEL_DIR, "xgb_model_4.json")
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")

    # ================================================================
    # Predictions and reconstruction
    # ================================================================
    preds_test_factor = bst.predict(dtest)
    eps = 1e-8
    denom = np.where(np.abs(lsum_test) < eps, eps, lsum_test)
    E_pred = (preds_test_factor / 100.0) * denom  # reconstructed energy
    E_true = ytrue_test

    # Compute metrics
    mse = mean_squared_error(E_true, E_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(E_true, E_pred)
    rel_errors = (E_pred - E_true) / np.where(np.abs(E_true) < eps, eps, E_true)
    mre = np.mean(rel_errors)
    mare = np.mean(np.abs(rel_errors))

    print("\n=== Test Set Metrics (on reconstructed E_pred) ===")
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
        preds_factor_r = bst.predict(dval, iteration_range=(0, r))
        E_pred_r = (preds_factor_r / 100.0) * np.where(np.abs(lsum_val) < eps, eps, lsum_val)
        rel = (E_pred_r - ytrue_val) / np.where(np.abs(ytrue_val) < eps, eps, ytrue_val)
        val_mre_list.append(np.mean(rel))
        val_mare_list.append(np.mean(np.abs(rel)))

    # ---- RMSE & MAE ----
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label="Train RMSE (factor)")
    plt.plot(epochs, val_rmse, label="Val RMSE (factor)")
    plt.plot(epochs, train_mae, "--", label="Train MAE (factor)")
    plt.plot(epochs, val_mae, "--", label="Val MAE (factor)")
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
    plt.ylabel("Relative Error (Reconstructed Energy)")
    plt.title("Validation Relative Errors vs Epoch")
    plt.legend()
    plt.grid(True)
    rel_path = os.path.join(FIGURES_DIR, "relative_error_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(rel_path, dpi=150)
    plt.close()
    print(f"Saved relative error plot: {rel_path}")

    # ================================================================
    # Feature Importance (Sequential Order)
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
