import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from Preprocessing import prepare_event_feature_tensors_gpu # GPU-accelerated preprocessing


# ==========================================================
# 1. Define the Neural Network Model
# ==========================================================
class EnergyRegressor(nn.Module):
    def __init__(self, input_dim):
        super(EnergyRegressor, self).__init__()
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

    def forward(self, x):
        return self.net(x)


# ==========================================================
# 2. Training Function
# ==========================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=50, device="cuda"):
    Loss_history = {"epochs": [], "train": [], "val": []}
    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        Loss_history["epochs"].append(epoch+1)
        Loss_history["train"].append(train_loss)
        Loss_history["val"].append(val_loss)
    print("Training complete.")
    return model, Loss_history


# ==========================================================
# 3. Evaluation Function
# ==========================================================
def evaluate_model(model, test_loader, criterion, device="cuda"):
    model.eval()
    test_loss = 0.0
    preds, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            preds.append(outputs.cpu())
            truths.append(y_batch.cpu())

    test_loss /= len(test_loader.dataset)
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()

    print(f"Final Test MSE: {test_loss:.6f}")
    return preds, truths, test_loss


# ==========================================================
# 4. Main Script
# ==========================================================
if __name__ == "__main__":
    # Step 1: Prepare Data (GPU Preprocessing)
    filename = "hgcal_electron_data_0001.h5"  # change if needed
    print("Preparing features using GPU preprocessing...")
    X_tensor, y_tensor = prepare_event_feature_tensors_gpu(filename, batch_size=20000)

    # Step 2: Split Dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)
    test_loader = DataLoader(test_set, batch_size=512)

    # Step 3: Initialize Model, Loss, Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnergyRegressor(X_tensor.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Step 4: Train Model
    model, Loss_history = train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=50, device=device)
    # Optional: Save Loss History
    import json
    with open("loss_history.json", "w") as f:
        json.dump(Loss_history, f)
    
    # Step 5: Evaluate on Test Set
    preds, truths, test_loss = evaluate_model(model, test_loader, criterion, device)

    # Step 6: Save Model
    import os
    with open("config.json", "r") as f:
        config = json.load(f)
        models_dir = config.get("models_dir", ".")
    torch.save(model.state_dict(), os.path.join(models_dir, "energy_regressor.pt"))
    print("Model saved as 'energy_regressor.pt'.")
