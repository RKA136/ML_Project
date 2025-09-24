# Function file 

import h5py, numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json 
import os 
import pandas as pd

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

def prepare_event_layer_dataframe(filename="hgcal_electron_data_0001.h5"):
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
