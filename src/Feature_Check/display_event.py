import h5py
import numpy as np
import plotly.graph_objects as go
import os
import json

def display_event(event_index, filename, config_file="config.json"):
    """
    Extract a single event from an HDF5 file and plot in 3D using Plotly,
    with semi-transparent planes at each unique z layer.
    
    Parameters:
    -----------
    event_index : int
        Index of the event to plot (0-based).
    filename : str
        Name of the HDF5 file containing HGCAL data.
    config_file : str
        JSON file containing 'data_dir' and 'figures_dir'.
    """
    # Load configuration
    with open(config_file, "r") as f:
        config = json.load(f)
    data_dir = config["data_dir"]
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    
    # Open HDF5 file
    with h5py.File(filepath, "r") as f:
        nhits = f["nhits"][:]
        xs = f["rechit_x"]
        ys = f["rechit_y"]
        zs = f["rechit_z"]
        energies = f["rechit_energy"]
        targets = f["target"][:]

        if event_index < 0 or event_index >= len(nhits):
            raise IndexError(f"Event index {event_index} out of range (0-{len(nhits)-1})")
        
        # Compute start and end indices for this event
        start = int(np.sum(nhits[:event_index]))
        end = int(np.sum(nhits[:event_index + 1]))
        
        # Load only the slice for this event
        x = np.array(xs[start:end])
        y = np.array(ys[start:end])
        z = np.array(zs[start:end])
        e = np.array(energies[start:end])
        true_energy = targets[event_index]

    # Determine the extent of the detector in x and y
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Create Plotly figure
    fig = go.Figure()

    # Add rechits
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=5, color=e, colorscale="Viridis", opacity=0.8,
                    colorbar=dict(title="Energy (MIP)")),
        text=[f"E={ee:.2f} MIP" for ee in e],
        name="Rechits"
    ))

    # Add semi-transparent planes for each unique z-layer
    unique_zs = np.unique(z)
    for layer_z in unique_zs:
        fig.add_trace(go.Mesh3d(
            x=[x_min, x_max, x_max, x_min],
            y=[y_min, y_min, y_max, y_max],
            z=[layer_z]*4,
            color="lightgray",
            opacity=0.2,
            name=f"Layer z={layer_z}",
            showlegend=False
        ))

    # Layout
    fig.update_layout(
        title=f"3D Shower with Layers (Event {event_index}, True E={true_energy:.1f} GeV)",
        scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show and save figure
    fig_path = os.path.join(figures_dir, f"event_{event_index}_layers.html")
    fig.show()
    fig.write_html(fig_path)
    print(f"Saved interactive figure as {fig_path}")


# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    display_event(event_index=200, filename="hgcal_electron_data_0001.h5")
