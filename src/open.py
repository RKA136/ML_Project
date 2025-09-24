import h5py, numpy as np
import plotly.graph_objects as go
import json 
import os 

# Load config.json
with open("config.json", "r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
figures_dir = config["figures_dir"]

# Make sure figures folder exists
os.makedirs(figures_dir, exist_ok=True)

dataset_path = os.path.join(data_dir, "hgcal_electron_data_0001.h5")

# Load one event
with h5py.File(dataset_path, "r") as f:
    nhits = f["nhits"][:]
    xs, ys, zs, energies = f["rechit_x"][:], f["rechit_y"][:], f["rechit_z"][:], f["rechit_energy"][:]
    targets = f["target"][:]

    i = 200   # choose event index
    start, end = int(np.sum(nhits[:i])), int(np.sum(nhits[:i+1]))
    x, y, z, e = xs[start:end], ys[start:end], zs[start:end], energies[start:end]
    true_E = targets[i]

# Normalize marker size for visibility
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
fig_path = os.path.join(figures_dir, "event_display.html")
fig.show()
fig.write_html(fig_path)
print(f"Saved interactive figure as {fig_path}")


num_events = len(nhits)

print(f"Total number of events: {num_events}")