# =============================
# XGBoost Regression Training (GPU) without E_sum and E_max
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
# Load preprocessed tensors
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)
data_dir = config["data_dir"]

data_path = os.path.join(data_dir, "processed_data_0001.pt")
data = torch.load(data_path)

X_tensor = data["X"]
y_tensor = data["y"]

# -----------------------------
# Convert to numpy arrays, drop E_sum and E_max
# -----------------------------
# Columns 0=E_sum, 1=E_max → remove both
X = np.delete(X_tensor.numpy(), [0, 1], axis=1)
y = y_tensor.numpy().ravel()  # flatten to 1D array

print(f"Dataset shape after dropping E_sum and E_max: X={X.shape}, y={y.shape}")

# -----------------------------
# Train/Validation Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} events")
print(f"Validation set: {X_val.shape[0]} events")

# -----------------------------
# Initialize XGBRegressor (GPU)
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
n_features = X.shape[1]
# Feature names: skip E_sum and E_max, keep r_std, z_std, r90 + all layer fractions
feature_names = ["r_std", "z_std", "r90"] + [f"E_layer_frac_{i}" for i in range(n_features - 3)]

plt.figure(figsize=(10,6))
plt.barh(range(n_features), model.feature_importances_)
plt.yticks(range(n_features), feature_names)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance (without E_sum and E_max)")
plt.tight_layout()
plt.show()

# -----------------------------
# Save the Trained Model
# -----------------------------
model_path = os.path.join(data_dir, "xgb_regressor_model_no_Esum_Emax.joblib")
joblib.dump(model, model_path)
print(f"Trained model saved at: {model_path}")
