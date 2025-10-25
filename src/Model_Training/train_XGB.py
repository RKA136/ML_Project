# =============================
# XGBoost Regression Training (GPU) — Using Vectorized Features
# =============================

import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load config and data
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
input_file = os.path.join(data_dir, "hgcal_electron_data_large_processed.pt")

print(f"Loading preprocessed data from: {input_file}")
data = torch.load(input_file)

X = data["data"].numpy()
y = data["targets"].numpy().ravel()

print(f"Loaded dataset: X={X.shape}, y={y.shape}")

# -----------------------------
# Feature naming (based on preprocessing)
# -----------------------------
n_layers = 28
feature_names = [f"layer_frac_{i}" for i in range(n_layers)] + [
    "r_cog", "r_k3", "r_k4", "r_k5",
    "z_cog", "z_k3", "z_k4", "z_k5"
]
assert len(feature_names) == X.shape[1], "Feature name count mismatch!"

# -----------------------------
# Train/Validation Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} events")
print(f"Validation set: {X_val.shape[0]} events")

# -----------------------------
# Initialize XGBoost Regressor (GPU)
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    random_state=42,
    verbosity=1,
    eval_metric="rmse",
    early_stopping_rounds=20
)

# -----------------------------
# Train Model with Early Stopping
# -----------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# -----------------------------
# Evaluate Performance
# -----------------------------
y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\nValidation Metrics:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")

# -----------------------------
# Feature Importance Plot
# -----------------------------
plt.figure(figsize=(10, 8))
sorted_idx = np.argsort(model.feature_importances_)[::-1]
plt.barh(
    np.array(feature_names)[sorted_idx],
    model.feature_importances_[sorted_idx]
)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance (Vectorized Features)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------
# Save Trained Model
# -----------------------------
model_path = os.path.join(data_dir, "xgb_regressor_vectorized.joblib")
joblib.dump(model, model_path)
print(f"\nTrained model saved at: {model_path}")
