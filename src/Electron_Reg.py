# Function file 

import h5py, numpy as np
import plotly.graph_objects as go
import json 
import os 

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