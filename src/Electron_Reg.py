# Function file 

import h5py, numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json 
import os 
import pandas as pd
import cupy as cp
from tqdm import tqdm

def load_data(filename="hgcal_electron_data_0001.h5"):
    """Load the HGCAL electron dataset from an HDF5 file.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    Returns:
        dict: A dictionary containing the dataset.
    """
    dataset = {}
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config["data_dir"]
    filename = os.path.join(data_dir, filename)
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            dataset[key] = f[key][:]
    return dataset

def display_event(event_index=200, filename="hgcal_electron_data_0001.h5"):
    """Display a 3D event from the HGCAL electron dataset.
    Args:
        event_index (int): Index of the event to display.
        filename (str): Name of the HDF5 file containing the dataset.
    Returns: html file with interactive 3D plot of the event.
    """
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
    sizes = 5 # scale marker sizes with energy
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=e,
            colorscale='Viridis',
            colorbar=dict(title="Energy (MIP)"),
            opacity=0.7
        ),
        text=[f"E={ee:.2f} MIP" for ee in e],  # hover text
    )])
    fig.update_layout(
        title=f"Interactive 3D Shower (Event {i}, True E = {true_E:.1f} GeV)",
        scene=dict(
            xaxis_title='x [cm]',
            yaxis_title='y [cm]',
            zaxis_title='z [cm]'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig_path = os.path.join(figures_dir, f"event_display_{i}.html")
    fig.show()
    fig.write_html(fig_path)
    print(f"Saved interactive figure as {fig_path}")

def hits_per_event(filename="hgcal_electron_data_0001.h5"):
    """Plot a histogram of the number of hits per event in the dataset.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    """
    dataset = load_data(filename)
    nhits = dataset["nhits"]
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nhits, bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Number of Hits per Event")   
    ax.set_xlabel("Number of Hits")
    ax.set_ylabel("Number of Events")
    hist_path = os.path.join(figures_dir, "hits_per_event.png")
    plt.savefig(hist_path)
    plt.close()

def true_energy_distribution(filename="hgcal_electron_data_0001.h5"):
    """Plot a histogram of the true energy distribution of events in the dataset.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    """
    dataset = load_data(filename)
    targets = dataset["target"]
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(targets, bins=30, color='lightgreen', edgecolor='black')
    ax.set_title("Histogram of True Energy Distribution")   
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Number of Events")
    hist_path = os.path.join(figures_dir, "true_energy_distribution.png")
    plt.savefig(hist_path)
    plt.close()

def prepare_event_layer_dataframe_cpu(filename="hgcal_electron_data_0001.h5"):
    """
    Prepare a DataFrame with average energy per layer for each event.

    Args:
        filename (str, optional): Name of the HDF5 file. Defaults to "hgcal_electron_data_0001.h5".

    Returns:
        pd.DataFrame: DataFrame with average energy per layer for each event.
    """
    dataset = load_data(filename)
    nhits = dataset["nhits"]
    zs, energies = dataset["rechit_z"], dataset["rechit_energy"]

    n_events = len(nhits)

    # Convert nhits to integer to avoid TypeError
    nhits = nhits.astype(int)

    # Unique sorted z-layers
    unique_zs = np.sort(np.unique(zs))
    n_layers = len(unique_zs)

    # Event indices for each hit
    event_indices = np.repeat(np.arange(n_events), nhits)

    # Map z-values to layer columns
    z_to_col = {z: i for i, z in enumerate(unique_zs)}
    col_indices = np.array([z_to_col[z] for z in zs])

    # Linear indices for bincount
    linear_idx = event_indices * n_layers + col_indices

    # Aggregate energies
    energy_sum = np.bincount(linear_idx, weights=energies, minlength=n_events * n_layers)
    hit_count = np.bincount(linear_idx, minlength=n_events * n_layers)

    avg_energy = energy_sum / np.maximum(hit_count, 1)
    avg_energy_matrix = avg_energy.reshape(n_events, n_layers)

    column_names = [f"z_{i+1}_average_energy" for i in range(n_layers)]
    df = pd.DataFrame(avg_energy_matrix, columns=column_names)
    df.insert(0, "event_no", np.arange(n_events))

    return df

def prepare_event_layer_dataframe_gpu(filename="hgcal_electron_data_0001.h5", batch_size=10000):
    dataset = load_data(filename)
    nhits = dataset["nhits"].astype(int)
    zs, energies = dataset["rechit_z"], dataset["rechit_energy"]

    n_events = len(nhits)
    print(f"Total events: {n_events}")

    # First pass: get all unique z-values (on CPU to avoid VRAM overload)
    unique_zs_cpu = np.unique(zs)
    n_layers = len(unique_zs_cpu)

    # Preallocate CPU matrix for results
    avg_energy_matrix_cpu = np.zeros((n_events, n_layers), dtype=np.float32)

    # Batch processing loop
    start = 0
    for batch_start in tqdm(range(0, n_events, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, n_events)

        # Slice batch
        nhits_batch = nhits[batch_start:batch_end]
        zs_batch = zs[start:start + nhits_batch.sum()]
        energies_batch = energies[start:start + nhits_batch.sum()]

        # Transfer to GPU
        nhits_gpu = cp.asarray(nhits_batch)
        zs_gpu = cp.asarray(zs_batch)
        energies_gpu = cp.asarray(energies_batch)

        # Event indices for this batch
        event_indices = cp.repeat(cp.arange(batch_end - batch_start), nhits_gpu)

        # Map z-values to layer indices (GPU search against CPU-computed unique_zs)
        unique_zs_gpu = cp.asarray(unique_zs_cpu)
        col_indices = cp.searchsorted(unique_zs_gpu, zs_gpu)

        # Linear indices
        linear_idx = event_indices * n_layers + col_indices

        # Bin counts
        energy_sum = cp.bincount(linear_idx, weights=energies_gpu, minlength=(batch_end - batch_start) * n_layers)
        hit_count = cp.bincount(linear_idx, minlength=(batch_end - batch_start) * n_layers)

        # Average energy
        avg_energy = energy_sum / cp.maximum(hit_count, 1)
        avg_energy_matrix_batch = avg_energy.reshape(batch_end - batch_start, n_layers)

        # Copy results to CPU
        avg_energy_matrix_cpu[batch_start:batch_end, :] = cp.asnumpy(avg_energy_matrix_batch)

        # Free GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        # Update offset into hit arrays
        start += nhits_batch.sum()

    # Build DataFrame
    column_names = [f"z_{i+1}_average_energy" for i in range(n_layers)]
    df = pd.DataFrame(avg_energy_matrix_cpu, columns=column_names)
    df.insert(0, "event_no", np.arange(n_events))

    return df

def plot_average_energy_per_layer(df):
    """Plot average energy per layer for a random sample of events.
    Args:
        df (pd.DataFrame): DataFrame containing average energy per layer for each event.
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    n_layers = df.shape[1] - 1  # Exclude event_no column
    layer_indices = np.arange(1, n_layers + 1)

    sample_events = df.sample(n=1, random_state=42)

    plt.figure(figsize=(10, 6))
    for _, row in sample_events.iterrows():
        plt.plot(layer_indices, row[1:], marker='o', label=f"Event {int(row['event_no'])}")

    plt.title("Average Energy per Layer for Sample Events")
    plt.xlabel("Layer (z index)")
    plt.ylabel("Average Energy (MIP)")
    plt.xticks(layer_indices)
    plt.legend()
    plt.grid()
    plot_path = os.path.join(figures_dir, "average_energy_per_layer_sample_events.png")
    plt.savefig(plot_path)
    plt.close()