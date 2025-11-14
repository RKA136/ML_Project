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
    mode5 → layerwise [N_hits, E1/E7, E7/E19]

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
import optuna

# ================================================================
# Configuration
# ================================================================
DATA_MODE = "mode4"          # choose among: mode1, mode2, mode3, mode4, mode5
MANUAL_DATA_PATH = None      # optional override

TEST_SIZE = 0.10
VAL_SIZE = 0.10
NUM_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
USE_GPU = True
VERBOSE_EVAL = 25

# Optuna settings
OPTUNA_TRIALS = 40           # number of optuna trials (adjustable)
OPTUNA_TIMEOUT = None        # in seconds, optional (set to int to limit time)


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
        "mode5": {
            "path": "processed_data_0001_v5.pt",
            "description": "v5: [N_hits (28), E1/E7 (28), E7/E19 (28)] — per-layer hit counts plus ring ratios"
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
# Optuna Hyperparameter Optimization
# ================================================================
def suggest_xgb_params(trial, use_gpu=True):
    """
    Extended XGBoost parameter search space suggested to Optuna trial.
    The space is intentionally broad to accommodate diverse calorimeter datasets.
    """
    # learning rate and tree params
    eta = trial.suggest_float("eta", 0.01, 0.20, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 12)
    min_child_weight = trial.suggest_float("min_child_weight", 1.0, 20.0)
    gamma = trial.suggest_float("gamma", 0.0, 10.0)
    max_delta_step = trial.suggest_int("max_delta_step", 0, 10)

    # regularization
    reg_lambda = trial.suggest_float("lambda", 1e-3, 10.0, log=True)
    reg_alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)

    # sampling
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
    colsample_bynode = trial.suggest_float("colsample_bynode", 0.5, 1.0)

    # tree growth and leaves
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    # max_leaves is required when using lossguide
    max_leaves = trial.suggest_int("max_leaves", 16, 512)

    # monotone constraints / interaction constraints (optional stubs)
    # For calorimeter variables these are typically not set; kept here for completeness.
    # monotone_constraints = trial.suggest_categorical("monotone_constraints", [None, "(1,0,-1)"])
    # interaction_constraints = trial.suggest_categorical("interaction_constraints", [None, "[]"])

    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": eta,
        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "gamma": float(gamma),
        "max_delta_step": int(max_delta_step),
        "lambda": float(reg_lambda),
        "alpha": float(reg_alpha),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "colsample_bylevel": float(colsample_bylevel),
        "colsample_bynode": float(colsample_bynode),
        "grow_policy": grow_policy,
        "max_leaves": int(max_leaves),
        "seed": RANDOM_STATE,
        "verbosity": 0,
    }

    if use_gpu:
        params["tree_method"] = "gpu_hist"
    else:
        params["tree_method"] = "hist"

    return params


def run_optuna_search(X_train, y_train, X_val, y_val, n_trials=40, timeout=None):
    """
    Runs an Optuna study to minimize validation RMSE.
    Returns best parameter dictionary (unprocessed).
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, "train"), (dval, "validation")]

    def objective(trial):
        params = suggest_xgb_params(trial, use_gpu=USE_GPU)
        evals_result = {}

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=watchlist,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=False,
        )

        # Determine the validation rmse at best_iteration (fallback to last if not available)
        if hasattr(bst, "best_iteration") and bst.best_iteration is not None:
            bi = int(bst.best_iteration)
            # evals_result structure: {"train": {"rmse": [...], "mae": [...]}, "validation": {...}}
            try:
                val_rmse = evals_result["validation"]["rmse"][bi - 1] if bi > 0 else evals_result["validation"]["rmse"][0]
            except Exception:
                # fallback to final value
                val_rmse = evals_result["validation"]["rmse"][-1]
        else:
            val_rmse = evals_result["validation"]["rmse"][-1]

        # Optuna minimizes the returned value
        return float(val_rmse)

    study = optuna.create_study(direction="minimize")
    if timeout is None:
        study.optimize(objective, n_trials=n_trials)
    else:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params.copy()

    # convert categorical/derived params to types expected by xgboost
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])
    if "max_leaves" in best_params:
        best_params["max_leaves"] = int(best_params["max_leaves"])
    if "max_delta_step" in best_params:
        best_params["max_delta_step"] = int(best_params["max_delta_step"])

    # Ensure that eval_metric and objective are present (they will be enforced later too)
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = ["rmse", "mae"]
    best_params["seed"] = RANDOM_STATE
    if USE_GPU:
        best_params["tree_method"] = "gpu_hist"
    else:
        best_params["tree_method"] = "hist"

    print("\n[INFO] Optuna optimization complete.")
    print(f"[INFO] Best validation RMSE: {study.best_value:.6e}")
    print("[INFO] Best params (sample):")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return best_params


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
    FIGURES_SUBDIR = os.path.join(FIGURES_ROOT, f"figures_{tag}_{TARGET_MODE_NAME}_optuna")
    MODELS_SUBDIR = os.path.join(MODELS_ROOT, f"model_{tag}_{TARGET_MODE_NAME}_optuna")
    os.makedirs(FIGURES_SUBDIR, exist_ok=True)
    os.makedirs(MODELS_SUBDIR, exist_ok=True)
    print(f"[INFO] Figures folder: {FIGURES_SUBDIR}")
    print(f"[INFO] Models folder : {MODELS_SUBDIR}")

    # ================================================================
    # Data Splitting
    # ================================================================
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_target, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_fraction, random_state=RANDOM_STATE)

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ================================================================
    # Hyperparameter Optimization via Optuna (ADD-ON)
    # ================================================================
    try:
        print(f"\n[INFO] Running Optuna hyperparameter search (trials={OPTUNA_TRIALS})...")
        best_params = run_optuna_search(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    except Exception as e:
        print(f"[WARNING] Optuna optimization failed or raised exception: {e}")
        print("[WARNING] Falling back to default parameter set.")
        best_params = {
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
            best_params["tree_method"] = "gpu_hist"

    # ================================================================
    # Prepare DMatrix objects
    # ================================================================
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}

    # ================================================================
    # Training with best parameters
    # ================================================================
    print(f"\n[INFO] Starting XGBoost training — {tag} ({TARGET_MODE_NAME}) with optimized parameters")
    # Ensure params are fully valid types for xgboost
    params = best_params.copy()
    # enforce types and minimal keys
    params["objective"] = params.get("objective", "reg:squarederror")
    params["eval_metric"] = params.get("eval_metric", ["rmse", "mae"])
    params["seed"] = params.get("seed", RANDOM_STATE)
    params["verbosity"] = params.get("verbosity", 1)
    if USE_GPU and "tree_method" not in params:
        params["tree_method"] = "gpu_hist"

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
    try:
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
    except Exception as e:
        print(f"[WARNING] Could not plot training curves: {e}")

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

        try:
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
        except Exception as e:
            print(f"[WARNING] Could not save feature importance plot ({imp_type}): {e}")

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
        "metrics_plot": metrics_path if 'metrics_path' in locals() else None,
        "feature_importance_json": importance_json_path,
        "figures_dir": FIGURES_SUBDIR,
        "models_dir": MODELS_SUBDIR,
        "xgboost_params": params,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved: {summary_path}")


# ================================================================
# Entrypoint
# ================================================================
if __name__ == "__main__":
    train()
