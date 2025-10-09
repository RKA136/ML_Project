import h5py
import numpy as np
import torch
import time
from Preprocessing import load_h5 
import os, json

with open("config.json", "r") as f:
    config = json.load(f)
data_dir = config["data_dir"]

filename = "hgcal_electron_data_large.h5"  # raw input file
preprocessed_file = "processed_data_large.pt"  # where tensors were saved

processed_filepath = os.path.join(data_dir, preprocessed_file)


print(f"Loading preprocessed tensors from {preprocessed_file}...")

data = torch.load(processed_filepath, map_location="cpu", weights_only=True)  # always use map_location="cpu" for safety

print("Type of data:", type(data))

if isinstance(data, dict):
    print("Keys in the file:", data.keys())
    for k, v in data.items():
        print(f"Key: {k}, type: {type(v)}, shape:", getattr(v, "shape", None))
else:
    print(data)

# Load preprocessed tensors (assumes they were saved with torch.save)
saved_data = torch.load(processed_filepath, map_location="cpu", weights_only=True)
X_tensor = saved_data["X"]
y_tensor = saved_data["y"]

print(f"X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")

# -----------------------------------------
# Manual feature computation for one event
# -----------------------------------------
with load_h5(filename) as data:
    nhits = data["nhits"][:]  # small array
    true_E_all = data["true_energy"][:]  # small array

    event_idx = 0  # choose first event
    start_idx = int(nhits[:event_idx].sum())
    end_idx = int(start_idx + nhits[event_idx])

    x_ev = data["rechit_x"][start_idx:end_idx]
    y_ev = data["rechit_y"][start_idx:end_idx]
    z_ev = data["rechit_z"][start_idx:end_idx]
    E_ev = data["rechit_energy"][start_idx:end_idx]

    E_sum = E_ev.sum()
    E_max = E_ev.max()
    x_cog = np.sum(x_ev * E_ev) / E_sum
    y_cog = np.sum(y_ev * E_ev) / E_sum
    z_cog = np.sum(z_ev * E_ev) / E_sum

    r = np.sqrt((x_ev - x_cog) ** 2 + (y_ev - y_cog) ** 2)
    r_std = np.sqrt(np.sum(E_ev * r**2) / E_sum - (np.sum(E_ev * r) / E_sum) ** 2)
    z_shift = z_ev - z_cog
    z_std = np.sqrt(np.sum(E_ev * z_shift**2) / E_sum - (np.sum(E_ev * z_shift) / E_sum) ** 2)

    order = np.argsort(r)
    cumE = np.cumsum(E_ev[order])
    idx90 = np.searchsorted(cumE, 0.9 * E_sum)
    r90 = r[order][min(idx90, len(r) - 1)]

    unique_zs = np.sort(np.unique(data["rechit_z"][:]))
    n_layers = len(unique_zs)
    layer_idx = np.searchsorted(unique_zs, z_ev)
    layer_idx = np.clip(layer_idx, 0, n_layers - 1)
    E_layer_sum = np.zeros(n_layers, dtype=np.float32)
    for li, e in zip(layer_idx, E_ev):
        E_layer_sum[li] += e
    E_layer_frac = E_layer_sum / E_sum

# ----------------------------------------------------------------------
# Compare features with loaded tensor
# ----------------------------------------------------------------------
features_manual = np.concatenate([[E_sum, E_max, r_std, z_std, r90], E_layer_frac])
X_event_1 = X_tensor[event_idx].numpy()

print("Manual features:", features_manual)
print("GPU (loaded) features:", X_event_1)

for i, (manual, gpu) in enumerate(zip(features_manual, X_event_1)):
    diff_1 = abs(manual - gpu)
    if diff_1 < 1e-3:
        print(f"Feature {i}: manual={manual:.6f}, GPU={gpu:.6f}, diff={diff_1:.6e} (MATCH)")
    else:
        print(f"Feature {i}: manual={manual:.6f}, GPU={gpu:.6f}, diff={diff_1:.6e} (MISMATCH)")
