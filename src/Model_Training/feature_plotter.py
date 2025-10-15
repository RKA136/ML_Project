import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def plot_feature_histograms(pt_filename="processed_data.pt", feature_names=None, bins=50, save_fig=False):
    """
    Plot histograms for first 5 features and scatter plot with errorbars for layer fractions.

    Args:
        pt_filename (str): Name of the saved .pt file.
        feature_names (list): Optional list of feature names for labeling.
        bins (int): Number of histogram bins for the first 5 features.
        save_fig (bool): If True, saves the figures in figures_dir.
    """
    # Load paths from config
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
        figures_dir = config["figures_dir"]

    # Load tensors
    pt_path = os.path.join(data_dir, pt_filename)
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    X = data["X"].numpy()
    y = data["y"].numpy()

    n_features = X.shape[1]

    # Default names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    # ---- Part 1: Plot histograms for first 5 features ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        ax = axes[i]
        ax.hist(X[:, i], bins=bins, color='steelblue', alpha=0.7, edgecolor='k')
        ax.set_title(feature_names[i], fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    fig.suptitle("Histograms of First 5 Features", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        out_path = os.path.join(figures_dir, "histograms_first5.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved first 5 feature histograms to: {out_path}")
    else:
        plt.show()

    # ---- Part 2: Scatter plot with error bars for layer energy fractions ----
    # Take remaining columns (layer fractions)
    layer_fracs = X[:, 5:]
    n_layers = layer_fracs.shape[1]

    # Compute mean and std per layer
    layer_means = np.mean(layer_fracs, axis=0)
    layer_stds = np.std(layer_fracs, axis=0)

    layer_indices = np.arange(1, n_layers + 1)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        layer_indices,
        layer_means,
        yerr=layer_stds,
        fmt='o',
        color='darkorange',
        ecolor='gray',
        elinewidth=1,
        capsize=4,
        markersize=6,
        alpha=0.8,
        label="Mean ± Std per layer"
    )

    plt.xlabel("Layer Number")
    plt.ylabel("Mean Layer Energy Fraction")
    plt.title("Average Energy Fraction per Layer with Error Bars")
    plt.grid(alpha=0.3)
    plt.legend()

    if save_fig:
        out_path2 = os.path.join(figures_dir, "layer_fraction_scatter.png")
        plt.savefig(out_path2, dpi=300)
        print(f"Saved layer scatter plot to: {out_path2}")
    else:
        plt.show()


if __name__ == "__main__":
    feature_names = [
        "E_sum", "E_max", "r_std", "z_std", "r90",
        *[f"Layer_{i}_frac" for i in range(1, 33)]  # assuming 32 layers
    ]

    plot_feature_histograms(
        "processed_data_large.pt",
        feature_names=feature_names,
        bins=100,
        save_fig=False
    )
