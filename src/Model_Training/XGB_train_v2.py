"""
xgb_train_and_evaluate.py

Usage: python xgb_train_and_evaluate.py

Produces:
 - figures/metrics_vs_epochs.png
 - figures/feature_importance.png
 - model/xgb_model.json (or .bin)
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

# -----------------------
# Configuration (tune as needed)
# -----------------------
NUM_ROUNDS = 200                # number of boosting rounds (epochs on x-axis)
EARLY_STOPPING_ROUNDS = 20
TEST_SIZE = 0.10                # holdout test set fraction
VAL_SIZE = 0.10                 # validation fraction (of remaining after test split)
RANDOM_STATE = 42
USE_GPU = False                 # set True to use 'gpu_hist' if XGBoost built with GPU
SCALE_FEATURES = False          # if True, standard scale features
MODEL_DIR = "model"
FIGURES_DIR = "figures"
VERBOSE_EVAL = 10               # print metrics every N rounds

# XGBoost parameters (regression)
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": ["rmse", "mae"],  # xgboost will supply these automatically
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": RANDOM_STATE,
    "verbosity": 1,
}
if USE_GPU:
    xgb_params["tree_method"] = "gpu_hist"

# -----------------------
# Helper evaluation metrics
# -----------------------
def mean_relative_error(preds, dtrain):
    """Custom evaluation metric: mean((pred - true)/true)"""
    labels = dtrain.get_label()
    # avoid division by zero: use small epsilon where label==0
    eps = 1e-8
    denom = np.where(np.abs(labels) < eps, eps, labels)
    rel = (preds - labels) / denom
    # return absolute and signed versions are both useful; here track signed MRE
    return "mre", float(np.mean(rel))

def mean_absolute_relative_error(preds, dtrain):
    labels = dtrain.get_label()
    eps = 1e-8
    denom = np.where(np.abs(labels) < eps, eps, labels)
    mare = np.mean(np.abs((preds - labels) / denom))
    return "mare", float(mare)

# -----------------------
# Utilities
# -----------------------
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------
# Load data (from your pipeline)
# -----------------------
def load_processed_tensors():
    # Reads config.json to find data_dir, then loads processed_data.pt
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    ppath = os.path.join(data_dir, "processed_data.pt")
    if not os.path.exists(ppath):
        raise FileNotFoundError(f"Processed data file not found: {ppath}")
    d = torch.load(ppath, map_location="cpu")
    X = d["X"].numpy() if isinstance(d["X"], torch.Tensor) else np.array(d["X"], dtype=np.float32)
    y = d["y"].numpy().reshape(-1) if isinstance(d["y"], torch.Tensor) else np.array(d["y"], dtype=np.float32).reshape(-1)
    return X, y

# -----------------------
# Training pipeline
# -----------------------
def train():
    ensure_dirs()
    print("Loading processed data...")
    X, y = load_processed_tensors()
    n_samples, n_features = X.shape
    print(f"Loaded {n_samples} samples x {n_features} features")

    # Train / test split (first carve out test), then train/val split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_frac_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_frac_of_temp, random_state=RANDOM_STATE)

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # Optional scaling (fit on training only)
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
    print("Starting training...")
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        feval=lambda preds, d: (mean_relative_error(preds, d), mean_absolute_relative_error(preds, d)),
        verbose_eval=VERBOSE_EVAL,
    )

    # Save model
    model_path = os.path.join(MODEL_DIR, "xgb_model.json")
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Predict on test set
    preds_test = bst.predict(dtest)
    # compute conventional metrics
    mse = mean_squared_error(y_test, preds_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds_test)
    # relative metrics
    eps = 1e-8
    denom = np.where(np.abs(y_test) < eps, eps, y_test)
    rel_errors = (preds_test - y_test) / denom
    mre = np.mean(rel_errors)           # signed
    mare = np.mean(np.abs(rel_errors))  # absolute

    print("Test set metrics:")
    print(f" MSE  : {mse:.6e}")
    print(f" RMSE : {rmse:.6e}")
    print(f" MAE  : {mae:.6e}")
    print(f" MRE  : {mre:.6e} (mean signed relative error)")
    print(f" MARE : {mare:.6e} (mean absolute relative error)")

    # Build metric arrays per epoch for plotting.
    # evals_result contains keys like 'train' -> 'rmse' list, 'validation' -> 'rmse' list.
    # Our custom metrics were added as fevals and will appear as 'train-mre'??? XGBoost doesn't automatically
    # include custom feval names in evals_result, so instead we will recompute per-round predictions
    # using staged_predict if available. Simpler: use evals_result for rmse/mae, and compute custom metrics
    # by re-running prediction on validation DMatrix using snapshots of trees up to each boosting round.
    # xgboost.Booster has predict iteration_range argument.

    # Prepare arrays
    train_rmse = evals_result.get("train", {}).get("rmse", [])
    val_rmse = evals_result.get("validation", {}).get("rmse", [])
    train_mae = evals_result.get("train", {}).get("mae", [])
    val_mae = evals_result.get("validation", {}).get("mae", [])

    rounds_trained = len(train_rmse)
    epochs = np.arange(1, rounds_trained + 1)

    # Compute MRE and MARE per epoch for validation by predicting with iteration_range up to each round
    val_mre_list = []
    val_mare_list = []
    for r in range(1, rounds_trained + 1):
        preds_r = bst.predict(dval, iteration_range=(0, r))
        labels = y_val
        denom = np.where(np.abs(labels) < eps, eps, labels)
        rel = (preds_r - labels) / denom
        val_mre_list.append(np.mean(rel))
        val_mare_list.append(np.mean(np.abs(rel)))

    # -----------------------
    # Plot metrics vs epochs
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label="train RMSE")
    plt.plot(epochs, val_rmse, label="val RMSE")
    plt.plot(epochs, train_mae, label="train MAE", linestyle="--")
    plt.plot(epochs, val_mae, label="val MAE", linestyle="--")
    plt.xlabel("Epoch (boosting round)")
    plt.ylabel("Error")
    plt.title("RMSE and MAE vs Epoch")
    plt.legend()
    plt.grid(True)
    metrics_fig_path = os.path.join(FIGURES_DIR, "metrics_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(metrics_fig_path, dpi=150)
    plt.close()
    print(f"Saved metrics plot to {metrics_fig_path}")

    # Plot validation relative errors (MRE, MARE)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_mre_list, label="val MRE (signed)")
    plt.plot(epochs, val_mare_list, label="val MARE (abs)", linestyle="--")
    plt.xlabel("Epoch (boosting round)")
    plt.ylabel("Relative Error")
    plt.title("Validation Relative Error vs Epoch")
    plt.legend()
    plt.grid(True)
    rel_fig_path = os.path.join(FIGURES_DIR, "relative_error_vs_epochs.png")
    plt.tight_layout()
    plt.savefig(rel_fig_path, dpi=150)
    plt.close()
    print(f"Saved relative-error plot to {rel_fig_path}")

    # -----------------------
    # Feature importance
    # -----------------------
    # Get feature scores and plot bar chart
    fmap = bst.get_score(importance_type="weight")
    # Convert to arrays (missing features will be absent)
    # xgboost feature names are 'f0','f1',...
    importances = np.zeros(n_features, dtype=float)
    for k, v in fmap.items():
        # k is like 'f12'
        if k.startswith("f"):
            idx = int(k[1:])
            if idx < n_features:
                importances[idx] = v
    # Normalize for readability
    if importances.sum() > 0:
        importances = importances / importances.sum()

    sorted_idx = np.argsort(importances)[::-1]
    topk = min(30, n_features)
    sorted_idx = sorted_idx[:topk]

    plt.figure(figsize=(10, max(4, 0.25 * topk)))
    labels = [f"f{i}" for i in sorted_idx]
    plt.barh(range(len(sorted_idx))[::-1], importances[sorted_idx])
    plt.yticks(range(len(sorted_idx))[::-1], labels)
    plt.xlabel("Normalized importance (weight)")
    plt.title("Feature importance (top features)")
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature importance to {fi_path}")

    # Also print top features to console
    print("Top feature importances (normalized):")
    for i in sorted_idx:
        print(f" f{i}: {importances[i]:.4f}")

    # Return a dict of key results for programmatic inspection if used as module
    results = {
        "model": bst,
        "evals_result": evals_result,
        "val_mre_list": val_mre_list,
        "val_mare_list": val_mare_list,
        "metrics_plot": metrics_fig_path,
        "rel_plot": rel_fig_path,
        "fi_plot": fi_path,
        "test_metrics": {"mse": mse, "rmse": rmse, "mae": mae, "mre": mre, "mare": mare},
    }
    return results

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    results = train()
    print("Training and evaluation completed.")
