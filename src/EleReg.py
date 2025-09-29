# hgcal_ml_dataset_lazy.py

import os
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cupy as cp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------
# 1. Dataset Loading Class
# --------------------------

class HGCALDataset(Dataset):
    """PyTorch Dataset for HGCAL events with lazy loading from HDF5."""

    def __init__(self, filename="hgcal_electron_data_0001.h5", transform=None, device="cpu"):
        self.filename = filename
        self.transform = transform
        self.device = device

        # Load config to get data directory
        with open("config.json", "r") as f:
            config = json.load(f)
        data_dir = config["data_dir"]
        self.filepath = os.path.join(data_dir, filename)

        # Open HDF5 file in read-only mode
        self.h5file = h5py.File(self.filepath, "r")

        # References to datasets (lazy, no data loaded yet)
        self.nhits = self.h5file["nhits"]
        self.xs = self.h5file["rechit_x"]
        self.ys = self.h5file["rechit_y"]
        self.zs = self.h5file["rechit_z"]
        self.energies = self.h5file["rechit_energy"]
        self.targets = self.h5file["target"]

        self.n_events = len(self.nhits)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        start = int(np.sum(self.nhits[:idx]))
        end = int(np.sum(self.nhits[:idx + 1]))

        x = torch.tensor(self.xs[start:end], dtype=torch.float32)
        y = torch.tensor(self.ys[start:end], dtype=torch.float32)
        z = torch.tensor(self.zs[start:end], dtype=torch.int32)
        e = torch.tensor(self.energies[start:end], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        sample = {"x": x, "y": y, "z": z, "energy": e, "target": target}

        if self.transform:
            sample = self.transform(sample)

        if self.device == "cuda":
            sample = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                      for k, v in sample.items()}

        return sample

    def close(self):
        """Close HDF5 file when done."""
        self.h5file.close()

# --------------------------
# 2. Data Loading Utility
# --------------------------

def load_data(filename="hgcal_electron_data_0001.h5"):
    """Load full dataset into memory (for plotting/EDA, optional)."""
    dataset = {}
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config["data_dir"]
    filepath = os.path.join(data_dir, filename)
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            dataset[key] = f[key][:]
    return dataset

# --------------------------
# 3. Visualization Utilities
# --------------------------

def display_event(event_index=200, filename="hgcal_electron_data_0001.h5"):
    """Interactive 3D event display."""
    dataset = load_data(filename)
    nhits = dataset["nhits"]
    xs, ys, zs, energies = dataset["rechit_x"], dataset["rechit_y"], dataset["rechit_z"], dataset["rechit_energy"]
    targets = dataset["target"]

    i = event_index
    start, end = int(np.sum(nhits[:i])), int(np.sum(nhits[:i+1]))
    x, y, z, e = xs[start:end], ys[start:end], zs[start:end], energies[start:end]
    true_E = targets[i]

    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=5, color=e, colorscale="Viridis", opacity=0.7,
                    colorbar=dict(title="Energy (MIP)")),
        text=[f"E={ee:.2f} MIP" for ee in e]
    )])
    fig.update_layout(
        title=f"3D Shower (Event {i}, True E={true_E:.1f} GeV)",
        scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig_path = os.path.join(figures_dir, f"event_display_{i}.html")
    fig.show()
    fig.write_html(fig_path)
    print(f"Saved interactive figure as {fig_path}")

def hits_per_event(filename="hgcal_electron_data_0001.h5"):
    """Histogram of number of hits per event."""
    dataset = load_data(filename)
    nhits = dataset["nhits"]

    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    plt.hist(nhits, bins=50, color="skyblue", edgecolor="black")
    plt.title("Number of Hits per Event")
    plt.xlabel("Number of Hits")
    plt.ylabel("Events")
    plt.savefig(os.path.join(figures_dir, "hits_per_event.png"))
    plt.close()

def true_energy_distribution(filename="hgcal_electron_data_0001.h5"):
    """Histogram of true event energy distribution."""
    dataset = load_data(filename)
    targets = dataset["target"]

    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    plt.hist(targets, bins=30, color="lightgreen", edgecolor="black")
    plt.title("True Energy Distribution")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Events")
    plt.savefig(os.path.join(figures_dir, "true_energy_distribution.png"))
    plt.close()

# --------------------------
# 4. Feature Engineering
# --------------------------

def prepare_event_layer_dataframe(filename="hgcal_electron_data_0001.h5", use_gpu=False, batch_size=10000):
    """
    Prepare a DataFrame with average energy per layer for each event, optionally using GPU in batches.

    Args:
        filename (str): HDF5 file with the dataset.
        use_gpu (bool): Whether to use GPU for computation.
        batch_size (int): Number of events per GPU batch.

    Returns:
        pd.DataFrame: Average energy per layer for each event.
    """
    import numpy as np
    import pandas as pd
    import cupy as cp
    from tqdm import tqdm
    import json
    import os

    # Load dataset
    dataset = load_data(filename)
    nhits = dataset["nhits"].astype(int)
    zs, energies = dataset["rechit_z"], dataset["rechit_energy"]
    n_events = len(nhits)

    # Unique sorted z-layers
    unique_zs = np.sort(np.unique(zs))
    n_layers = len(unique_zs)

    if not use_gpu:
        # ------------------------
        # CPU version
        # ------------------------
        event_indices = np.repeat(np.arange(n_events), nhits)
        z_to_col = {z: i for i, z in enumerate(unique_zs)}
        col_indices = np.array([z_to_col[z] for z in zs])
        linear_idx = event_indices * n_layers + col_indices

        energy_sum = np.bincount(linear_idx, weights=energies, minlength=n_events * n_layers)
        hit_count = np.bincount(linear_idx, minlength=n_events * n_layers)

        avg_energy = energy_sum / np.maximum(hit_count, 1)
        avg_energy_matrix = avg_energy.reshape(n_events, n_layers)

    else:
        # ------------------------
        # GPU version with batch processing
        # ------------------------
        avg_energy_matrix = np.zeros((n_events, n_layers), dtype=np.float32)
        start = 0

        for batch_start in tqdm(range(0, n_events, batch_size), desc="GPU batches"):
            batch_end = min(batch_start + batch_size, n_events)
            nhits_batch = nhits[batch_start:batch_end]
            nhits_cumsum = np.cumsum(nhits_batch)
            batch_start_idx = start
            batch_end_idx = start + nhits_batch.sum()

            zs_batch = zs[batch_start_idx:batch_end_idx]
            energies_batch = energies[batch_start_idx:batch_end_idx]

            # Transfer to GPU
            zs_gpu = cp.asarray(zs_batch)
            energies_gpu = cp.asarray(energies_batch)

            # Create event indices using Python list to avoid cp.repeat issues
            event_indices_list = []
            for i, n in enumerate(nhits_batch):
                event_indices_list.extend([i] * n)
            event_indices = cp.asarray(event_indices_list)

            # Map z-values to layer indices
            col_indices = cp.searchsorted(cp.asarray(unique_zs), zs_gpu)

            # Linear indices
            linear_idx = event_indices * n_layers + col_indices

            # Bin counts
            energy_sum = cp.bincount(linear_idx, weights=energies_gpu, minlength=(batch_end - batch_start) * n_layers)
            hit_count = cp.bincount(linear_idx, minlength=(batch_end - batch_start) * n_layers)

            # Average energy
            avg_energy = energy_sum / cp.maximum(hit_count, 1)
            avg_energy_matrix_batch = avg_energy.reshape(batch_end - batch_start, n_layers)

            # Copy results to CPU
            avg_energy_matrix[batch_start:batch_end, :] = cp.asnumpy(avg_energy_matrix_batch)

            # Free GPU memory
            cp.get_default_memory_pool().free_all_blocks()

            start += nhits_batch.sum()

    # Build DataFrame
    column_names = [f"z_{i+1}_average_energy" for i in range(n_layers)]
    df = pd.DataFrame(avg_energy_matrix, columns=column_names)
    df.insert(0, "event_no", np.arange(n_events))

    return df

def plot_average_energy_per_layer(df):
    """Line plot of average energy per layer for a sample event."""
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    n_layers = df.shape[1] - 1
    layer_indices = np.arange(1, n_layers + 1)

    sample_event = df.sample(n=1, random_state=42)

    plt.figure(figsize=(10, 6))
    for _, row in sample_event.iterrows():
        plt.plot(layer_indices, row[1:], marker="o", label=f"Event {int(row['event_no'])}")

    plt.title("Average Energy per Layer (Sample Event)")
    plt.xlabel("Layer index")
    plt.ylabel("Average Energy (MIP)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_dir, "average_energy_per_layer.png"))
    plt.close()
    
def plot_average_energy_per_layer_summary(df, save_name):
    """
    Plot the mean (and optional std) of average energy per layer across all events.

    Args:
        df (pd.DataFrame): DataFrame returned by prepare_event_layer_dataframe().
                           Must have columns ['event_no', 'z_1_average_energy', ...].
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os

    # Extract layer columns (exclude 'event_no')
    layer_cols = df.columns[1:]
    n_layers = len(layer_cols)
    layer_indices = np.arange(1, n_layers + 1)

    # Compute mean and standard deviation across events
    avg_energy_per_layer = df[layer_cols].mean(axis=0).values
    std_energy_per_layer = df[layer_cols].std(axis=0).values

    # Load figures directory from config
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Plot mean ± std
    plt.figure(figsize=(10, 6))
    plt.errorbar(layer_indices, avg_energy_per_layer, yerr=std_energy_per_layer,
                 fmt='o-', capsize=4, color='blue', ecolor='gray', elinewidth=1.5, markerfacecolor='orange')
    plt.title("Mean Average Energy per Layer Across All Events")
    plt.xlabel("Layer (z index)")
    plt.ylabel("Average Energy (MIP)")
    plt.xticks(layer_indices)
    plt.grid(True)
    plot_path = os.path.join(figures_dir, save_name)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved summary plot as {plot_path}")

