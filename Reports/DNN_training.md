# Deep Neural Network Regression Report

## 1. Objective

The goal of this project is to develop a **Deep Neural Network (DNN) regression model** to predict a continuous target variable `y` from a set of input features `X`. The model is implemented using PyTorch and evaluated using standard regression metrics.

---

## 2. Data Preparation

- **Dataset**: Preprocessed tensor data (`processed_data_0001.pt`) containing:
  - `X` → Feature matrix
  - `y` → Target vector
- **Feature Scaling**: Standardized using `StandardScaler` to zero mean and unit variance.
- **Train/Validation Split**: 80% training, 20% validation.
- **Batching**: `DataLoader` with batch size = 1024.

---

## 3. Model Architecture

The DNN model is a fully connected feedforward network with the following structure:

| Layer Type | Input Dim | Output Dim | Activation |
|------------|-----------|------------|------------|
| Linear     | Input Dim | 128        | ReLU       |
| Linear     | 128       | 64         | ReLU       |
| Linear     | 64        | 32         | ReLU       |
| Linear     | 32        | 1          | None       |

- **Input layer**: Matches the number of features in `X`.
- **Hidden layers**: Three layers with ReLU activation to introduce nonlinearity.
- **Output layer**: Single neuron for regression output.

---

## 4. Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW with learning rate = 1e-3
- **Device**: GPU if available, otherwise CPU
- **Epochs**: 50
- **Early Stopping**: Patience = 20 epochs based on validation MSE
- **Checkpoint**: Best model saved as `DNN_model.pt` in the `models_dir`

---

## 5. Training Summary

Validation MSE across epochs showed rapid decrease initially, followed by slow convergence:

- Epoch 1: Val MSE = 92.867
- Epoch 2: Val MSE = 21.115
- Epoch 3: Val MSE = 15.345
- Epoch 4: Val MSE = 12.997
- Epoch 5: Val MSE = 12.128
- Epoch 10: Val MSE = 9.968
- Epoch 20: Val MSE = 9.789
- Epoch 30: Val MSE = 10.678
- Epoch 40: Val MSE = 9.258
- Epoch 50: Val MSE = 9.118

- Rapid decrease in early epochs indicates effective learning.
- Small fluctuations in later epochs are typical for mini-batch updates.
- Early stopping was monitored but did not trigger due to occasional improvements.

---

## 6. Model Evaluation

The final model was evaluated on the validation set using the following metrics:

- MAE = 2.246
- RMSE = 3.005
- R² = 0.999

**Interpretation**:

- MAE = 2.246: Average prediction error is approximately 2.25 units.
- RMSE = 3.005: Root mean squared error slightly higher than MAE, indicating rare larger deviations.
- R² = 0.999: The model explains 99.9% of the variance, indicating an excellent fit.

---

## 7. Observations

- The DNN regression effectively captured the mapping from features `X` to target `y`.
- Minor fluctuations in validation loss in later epochs are normal.
- Future improvements may include:
  - Using a learning rate scheduler for smoother convergence.
  - Monitoring MAE or RMSE for early stopping instead of MSE.
  - Further hyperparameter tuning (layer size, activation, batch size).

---

## 8. Conclusion

The DNN model achieved **highly accurate predictions** with minimal error, demonstrating the effectiveness of deep learning for regression tasks on the given dataset. The architecture and training strategy successfully balanced convergence speed and generalization to unseen data.
