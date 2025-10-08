import h5py
import numpy as np
import time
from Preprocessing import prepare_event_feature_tensors_gpu, load_h5  # your function

filename = "hgcal_electron_data_0001.h5"  # example

# Generate full tensors from function using GPU
start_time_1 = time.time()
X_tensor, y_tensor = prepare_event_feature_tensors_gpu(filename, batch_size=20000)
stop_time_1 = time.time()
print(f"Processed file generated in {stop_time_1 - start_time_1} sec")

# Load raw data for manual check
data = load_h5(filename)
nhits = data["nhits"]
x_all, y_all, z_all, E_all = data["rechit_x"], data["rechit_y"], data["rechit_z"], data["rechit_energy"]
true_E_all = data["target"]

# Choose an event to test (e.g., first event)
event_idx = 0
start_idx = int(nhits[:event_idx].sum())
end_idx = int(start_idx + nhits[event_idx])

x_ev = x_all[start_idx:end_idx]
y_ev = y_all[start_idx:end_idx]
z_ev = z_all[start_idx:end_idx]
E_ev = E_all[start_idx:end_idx]

# Manually calculate features
E_sum = E_ev.sum()
E_max = E_ev.max()
x_cog = np.sum(x_ev * E_ev) / E_sum
y_cog = np.sum(y_ev * E_ev) / E_sum
z_cog = np.sum(z_ev * E_ev) / E_sum

r = np.sqrt((x_ev - x_cog)**2 + (y_ev - y_cog)**2)
r_std = np.sqrt(np.sum(E_ev * r**2) / E_sum - (np.sum(E_ev * r)/E_sum)**2)
z_shift = z_ev - z_cog
z_std = np.sqrt(np.sum(E_ev * z_shift**2)/E_sum - (np.sum(E_ev * z_shift)/E_sum)**2)

# r90 (energy containment radius)
order = np.argsort(r)
cumE = np.cumsum(E_ev[order])
idx90 = np.searchsorted(cumE, 0.9 * E_sum)
r90 = r[order][min(idx90, len(r)-1)]

# Energy fraction per layer
unique_zs = np.sort(np.unique(z_all))
n_layers = len(unique_zs)
layer_idx = np.searchsorted(unique_zs, z_ev)
layer_idx = np.clip(layer_idx, 0, n_layers-1)
E_layer_sum = np.zeros(n_layers, dtype=np.float32)
for li, e in zip(layer_idx, E_ev):
    E_layer_sum[li] += e
E_layer_frac = E_layer_sum / E_sum

# Stack features
features_manual = np.concatenate([[E_sum, E_max, r_std, z_std, r90], E_layer_frac])

# Compare with full X_tensor in gpu
X_event_1 = X_tensor[event_idx].numpy()

print("Manual features:", features_manual)
print("GPU function features:", X_event_1)

# Compare each component in GPU function
for i, (manual, gpu) in enumerate(zip(features_manual, X_event_1)):
    diff_1 = abs(manual - gpu)
    if diff_1 < 1e-3:
        print(f"Feature {i}: manual={manual:.6f}, GPU={gpu:.6f}, diff={diff_1:.6e} (MATCH)")
    else:
        print(f"FeatSure {i}: manual={manual:.6f}, GPU={gpu:.6f}, diff={diff_1:.6e} (MISMATCH)")