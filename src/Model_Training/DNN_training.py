#!/usr/bin/env python3
"""
DNN_training.py
-----------------------------------
This script trains a deep neural network (DNN) regressor in PyTorch to predict
calorimeter energy values from preprocessed feature tensors. It implements a
complete workflow from data loading to model evaluation, including normalization,
training with early stopping, and metric computation.

Pipeline Overview:
1. Load preprocessed feature and target tensors from `processed_data_0001.pt`.
2. Apply feature normalization using `StandardScaler` for stable optimization.
3. Split dataset into training and validation subsets (80%-20%).
4. Define a fully connected DNN architecture:
       Input → 128 → 64 → 32 → 1 (with ReLU activations)
5. Train the network using AdamW optimizer and MSE loss with early stopping.
6. Save the model when validation MSE improves, up to a patience threshold.
7. Reload the best checkpoint and evaluate performance using MAE, RMSE, and R².

The trained model is saved to:
   {models_dir}/DNN_model.pt

This framework provides a clean baseline for comparing deep learning regression
performance against traditional models (e.g., XGBoost).
"""

# =============================
# DNN Regression Training (PyTorch)
# =============================

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# Load preprocessed tensors
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)
data_dir = config["data_dir"]
model_dir = config["models_dir"]

data_path = os.path.join(data_dir, "processed_data_0001.pt")
data = torch.load(data_path)
X_tensor = data["X"]
y_tensor = data["y"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tensor.numpy())
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = y_tensor.float()

# -----------------------------
# Dataset & DataLoader
# -----------------------------
dataset = TensorDataset(X_scaled, y_tensor)
n_total = len(dataset)
n_val = int(0.2 * n_total)
n_train = n_total - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)

# -----------------------------
# Define the DNN Model
# -----------------------------
class DNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DNNRegressor(input_dim=X_scaled.shape[1]).to(device)

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# -----------------------------
# Training Loop with Early Stopping
# -----------------------------
best_val_loss = float("inf")
patience = 20
trigger_times = 0
n_epochs = 50

for epoch in range(n_epochs):
    # --- Training ---
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_losses.append(criterion(y_val_pred, y_val_batch).item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{n_epochs}, Val MSE: {val_loss:.6f}")

    # --- Early Stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), os.path.join(model_dir, "DNN_model.pt"))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# -----------------------------
# Load Best Model and Evaluate
# -----------------------------
model.load_state_dict(torch.load(os.path.join(model_dir, "DNN_model.pt")))
model.eval()

X_val_full = X_scaled[n_train:].to(device)
y_val_full = y_tensor[n_train:].to(device)
with torch.no_grad():
    y_pred = model(X_val_full).cpu().numpy()
y_val_full = y_val_full.cpu().numpy()

mae = mean_absolute_error(y_val_full, y_pred)
rmse = np.sqrt(mean_squared_error(y_val_full, y_pred))
r2 = r2_score(y_val_full, y_pred)

print("\nValidation Metrics:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")
