# Energy Regressor Neural Network — Training Report

## Overview
This script trains a neural network regressor to predict event energy values from processed detector features.  
The dataset is preprocessed on the GPU using the custom function `prepare_event_feature_tensors_gpu()` from the `Preprocessing` module.  

Training is implemented using PyTorch, with modularized functions for training, validation, and testing.  
The trained model is saved as `energy_regressor.pt` after training, along with the loss evolution in `loss_history.json`.

---

## 1. Model Architecture

### EnergyRegressor
A fully connected feedforward neural network for regression tasks.

```
self.net = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

**Key design choices:**
- ReLU activations for non-linearity.  
- Batch Normalization layers to stabilize training.  
- Final layer outputs a single continuous value — the predicted energy.

---

## 2. Data Preparation

### GPU-accelerated feature extraction
```
X_tensor, y_tensor = prepare_event_feature_tensors_gpu(filename, batch_size=20000)
```
- Loads data from an HDF5 file (`hgcal_electron_data_0001.h5`).  
- Computes event-level feature tensors on the GPU for efficiency.  

### Dataset Splitting
The dataset is split as:
- 80% → Training  
- 10% → Validation  
- 10% → Testing  

Implemented via:
```
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
```

### DataLoaders
```
train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
val_loader = DataLoader(val_set, batch_size=512)
test_loader = DataLoader(test_set, batch_size=512)
```

---

## 3. Training Process

### Function: `train_model()`

Handles the full training loop:
1. Forward pass through the model.  
2. MSE loss computation.  
3. Backward propagation and optimizer step.  
4. Evaluation on validation data at each epoch.

**Loss function:** `nn.MSELoss()` — Mean Squared Error, suitable for regression.  

**Optimizer:** `Adam(lr=1e-3)` — adaptive learning with stability.  

Outputs:
- Per-epoch train and validation losses.  
- JSON file `loss_history.json` storing the loss evolution.  

Example console output:
```
Epoch 50/50 | Train Loss: 23.755307 | Val Loss: 43.449352
```

---

## 4. Model Evaluation

### Function: `evaluate_model()`
After training, the model is evaluated on the test set:

```
Final Test MSE: 43.859067
```

Outputs:
- Predictions (`preds`)  
- Ground truth values (`truths`)  
- Final test loss  

These arrays are later used for generating diagnostic plots.

---

## 5. Saving Results

- **Trained Model:** `torch.save(model.state_dict(), "energy_regressor.pt")`  
- **Loss History:** Stored as `loss_history.json` for later plotting or analysis.

---

## Summary

| Component | Description |
|-----------|-------------|
| Framework | PyTorch |
| Model Type | Fully Connected Regressor |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam (lr=1e-3) |
| Hardware | GPU-supported (CUDA) |
| Training Duration | 50 epochs |
| Final Validation Loss | ≈ 43.45 |
| Final Test Loss (MSE) | ≈ 43.86 |
| Saved Artifacts | `energy_regressor.pt`, `loss_history.json` |

---
