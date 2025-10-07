# plotter.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from train_model import EnergyRegressor

def plot_model_results(model_class, model_file, test_data_file, input_dim=33, config_file="config.json"):
    """
    Generate evaluation plots for a trained model.
    
    Parameters:
    - model_class: callable, the class of the model (e.g., EnergyRegressor)
    - model_file: str, path to the saved model weights (.pt)
    - test_data_file: str, path to the test data (.pt) containing {'X_test', 'y_test'}
    - input_dim: int, input feature dimension
    - config_file: str, path to config.json containing 'figures_dir' and 'data_dir'
    """
    # Load config
    with open(config_file, "r") as f:
        config = json.load(f)
    figures_dir = config.get("figures_dir", "./figures")
    data_dir = config["data_dir"]

    # Create folder for this model's plots
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    model_figures_dir = os.path.join(figures_dir, model_name)
    os.makedirs(model_figures_dir, exist_ok=True)

    # Load model
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    # Load test data
    data = torch.load(test_data_file)
    X_test = data["X"]
    y_test = data["y"]

    # Inference
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    true = y_test.cpu().numpy().flatten()

    # Metrics
    mse = mean_squared_error(true, preds)
    r2 = r2_score(true, preds)
    residuals = preds - true
    abs_errors = np.abs(residuals)

    print(f"Test MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # -----------------------------
    # Plot 1 — Predicted vs True
    # -----------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(true, preds, alpha=0.5)
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title(f'Predicted vs True Energy (R² = {r2:.3f})')
    plt.savefig(os.path.join(model_figures_dir, "pred_vs_true.png"))
    plt.close()

    # -----------------------------
    # Plot 2 — Residual Distribution
    # -----------------------------
    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual (Pred - True)')
    plt.title('Residual Distribution')
    plt.savefig(os.path.join(model_figures_dir, "residual_distribution.png"))
    plt.close()

    # -----------------------------
    # Plot 3 — Residuals vs True
    # -----------------------------
    plt.figure()
    plt.scatter(true, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Energy')
    plt.ylabel('Residual (Pred - True)')
    plt.title('Residuals vs True Energy')
    plt.savefig(os.path.join(model_figures_dir, "residuals_vs_true.png"))
    plt.close()

    # -----------------------------
    # Plot 4 — Absolute Error vs True
    # -----------------------------
    plt.figure()
    plt.scatter(true, abs_errors, alpha=0.5)
    plt.xlabel('True Energy')
    plt.ylabel('|Prediction Error|')
    plt.title('Absolute Error vs True Energy')
    plt.savefig(os.path.join(model_figures_dir, "abs_error_vs_true.png"))
    plt.close()

    # -----------------------------
    # Plot 5 — Cumulative Error Distribution
    # -----------------------------
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(len(sorted_errors)) / len(sorted_errors)
    plt.figure()
    plt.plot(sorted_errors, cdf)
    plt.xlabel('|Prediction Error|')
    plt.ylabel('Fraction of Events')
    plt.title('Cumulative Error Distribution')
    plt.savefig(os.path.join(model_figures_dir, "error_cdf.png"))
    plt.close()

    print(f"All plots saved in '{model_figures_dir}/'")


if __name__ == "__main__":
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    data_dir = config["data_dir"]
    model_dir = config.get("models_dir", ".")

    # Paths to model weights and test data
    model_file = os.path.join(model_dir, "energy_regressor.pt")
    test_data_file = os.path.join(data_dir, "processed_data.pt")  # Ensure this file exists

    plot_model_results(
        model_class=EnergyRegressor,
        model_file=model_file,
        test_data_file=test_data_file,
        input_dim=33,
        config_file="config.json"
    )
