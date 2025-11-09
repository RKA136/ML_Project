#!/usr/bin/env python3
"""
preprocessing_v2.py
-----------------------------------
This script performs **fully vectorized CPU-based feature extraction** from large
HDF5 calorimeter datasets (e.g., CMS HGCAL simulation data) and saves the processed
data in PyTorch tensor format for subsequent machine learning workflows.

Overview:
---------
1. **Weighted Moment and Cumulant Computation**
   - Computes event-wise statistical moments (mean, skewness, kurtosis, etc.)
     using hit-level energy weighting.
   - Supports cumulant orders up to 5 for both radial (r) and longitudinal (z) hit distributions.

2. **Layer Energy Fractions**
   - Calculates per-layer total deposited energy and normalizes it to obtain
     fractional energy deposition per calorimeter layer.

3. **Vectorized Event Processing**
   - Processes all hits and events using batched NumPy operations.
   - Avoids event-by-event loops wherever possible, ensuring scalability to large datasets.

4. **Feature Set Generated per Event**
   - Layer-wise energy fractions (n_layers)
   - Radial cumulants: r_cog, r_k3, r_k4, r_k5
   - Longitudinal cumulants: z_cog, z_k3, z_k4, z_k5

   Total number of features per event:
       n_layers + 8

5. **Batch-wise Streaming**
   - Uses `batch_size` parameter to control memory usage.
   - Efficiently constructs padded hit arrays for vectorized cumulant computation.

6. **Data Output**
   - Saves results in a `.pt` PyTorch file containing:
       {
         "data": torch.Tensor(features),
         "targets": torch.Tensor(true_energy)
       }

Usage Example:
--------------
    python compute_features_vectorized.py

Configuration:
--------------
- Reads input and output directories from `config.json`.
- Input HDF5:   hgcal_electron_data_large.h5
- Output PyTorch: hgcal_electron_data_large_processed.pt
"""

import json
import os
import h5py
import numpy as np
import torch
import gc
from tqdm import tqdm

# -----------------------------
# Weighted cumulants (vectorized per event)
# -----------------------------
def weighted_cumulants(values, weights, order=3):
    """
    Compute weighted cumulants (skewness, kurtosis, higher orders) for each event.

    Args:
        values: 2D NumPy array (n_events × n_hits_per_event)
        weights: 2D NumPy array (same shape as values)
        order: int, cumulant order (3, 4, or 5)
    Returns:
        1D NumPy array of cumulant values per event
    """
    weights_sum = np.sum(weights, axis=1)
    mean = np.sum(weights * values, axis=1) / weights_sum
    centered = values - mean[:, None]

    if order == 3:
        return np.sum(weights * centered**3, axis=1) / weights_sum
    elif order == 4:
        var = np.sum(weights * centered**2, axis=1) / weights_sum
        return np.sum(weights * centered**4, axis=1) / weights_sum - 3 * var**2
    elif order == 5:
        var = np.sum(weights * centered**2, axis=1) / weights_sum
        skew = np.sum(weights * centered**3, axis=1) / weights_sum
        return np.sum(weights * centered**5, axis=1) / weights_sum - 10 * skew * var
    else:
        raise ValueError("Order must be 3, 4, or 5")

# -----------------------------
# Compute features fully vectorized
# -----------------------------
def compute_and_save_features_vectorized(h5_path, output_path, n_layers=28, batch_size=1000):
    print("Loading HDF5 file...")
    h5_file = h5py.File(h5_path, "r")
    nhits_all = np.array(h5_file["nhits"], dtype=np.int32)
    n_events = len(nhits_all)
    targets_all = np.array(h5_file["true_energy"], dtype=np.float32)

    x_all = np.array(h5_file["rechit_x"], dtype=np.float32)
    y_all = np.array(h5_file["rechit_y"], dtype=np.float32)
    z_all = np.array(h5_file["rechit_z"], dtype=np.float32)
    energy_all = np.array(h5_file["rechit_energy"], dtype=np.float32)

    features_list = []

    print("Processing events in batches (fully vectorized)...")
    for start_idx in tqdm(range(0, n_events, batch_size), desc="Batches processed"):
        end_idx = min(start_idx + batch_size, n_events)
        nhits_batch = nhits_all[start_idx:end_idx]

        # Compute flat array indices
        hit_start = int(np.sum(nhits_all[:start_idx])) if start_idx > 0 else 0
        hit_end = hit_start + int(np.sum(nhits_batch))

        x_batch_flat = x_all[hit_start:hit_end]
        y_batch_flat = y_all[hit_start:hit_end]
        z_batch_flat = z_all[hit_start:hit_end]
        e_batch_flat = energy_all[hit_start:hit_end]

        # Build event indices
        event_idx = np.repeat(np.arange(end_idx - start_idx), nhits_batch)

        # --- Layer fractions ---
        unique_layers = np.sort(np.unique(z_batch_flat))[:n_layers]
        layer_energy_batch = np.zeros((end_idx - start_idx, n_layers), dtype=np.float32)

        for i, layer in enumerate(unique_layers):
            mask = (z_batch_flat == layer)
            np.add.at(layer_energy_batch[:, i], event_idx[mask], e_batch_flat[mask])

        # Normalize to get fractions
        layer_frac_batch = layer_energy_batch / np.sum(layer_energy_batch, axis=1, keepdims=True)
        layer_frac_batch[np.isnan(layer_frac_batch)] = 0

        # --- Radial and Z values ---
        r_batch_flat = np.sqrt(x_batch_flat**2 + y_batch_flat**2)
        z_batch_flat = z_batch_flat  # already

        # Create padded arrays for vectorized cumulants
        max_hits = np.max(nhits_batch)
        n_events_batch = end_idx - start_idx

        r_padded = np.zeros((n_events_batch, max_hits), dtype=np.float32)
        z_padded = np.zeros((n_events_batch, max_hits), dtype=np.float32)
        e_padded = np.zeros((n_events_batch, max_hits), dtype=np.float32)

        # Fill padded arrays
        offsets = np.cumsum(np.insert(nhits_batch, 0, 0))
        for i in range(n_events_batch):
            start = offsets[i]
            end = offsets[i + 1]
            n = end - start
            r_padded[i, :n] = r_batch_flat[start:end]
            z_padded[i, :n] = z_batch_flat[start:end]
            e_padded[i, :n] = e_batch_flat[start:end]

        # --- Compute weighted features ---
        r_cog = np.sum(e_padded * r_padded, axis=1) / np.sum(e_padded, axis=1)
        r_k3 = weighted_cumulants(r_padded, e_padded, 3)
        r_k4 = weighted_cumulants(r_padded, e_padded, 4)
        r_k5 = weighted_cumulants(r_padded, e_padded, 5)

        z_cog = np.sum(e_padded * z_padded, axis=1) / np.sum(e_padded, axis=1)
        z_k3 = weighted_cumulants(z_padded, e_padded, 3)
        z_k4 = weighted_cumulants(z_padded, e_padded, 4)
        z_k5 = weighted_cumulants(z_padded, e_padded, 5)

        # Concatenate all features
        batch_features = np.hstack([
            layer_frac_batch,
            np.stack([r_cog, r_k3, r_k4, r_k5], axis=1),
            np.stack([z_cog, z_k3, z_k4, z_k5], axis=1)
        ])

        features_list.append(batch_features)
        gc.collect()

    # Convert to tensor and save
    features_tensor = torch.tensor(np.vstack(features_list), dtype=torch.float32)
    targets_tensor = torch.tensor(targets_all, dtype=torch.float32)
    torch.save({'data': features_tensor, 'targets': targets_tensor}, output_path)

    print(f"Processed features saved to {output_path}")
    h5_file.close()


# -----------------------------
# Main execution using config.json
# -----------------------------
if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    data_dir = config["data_dir"]
    input_file = os.path.join(data_dir, "hgcal_electron_data_large.h5")
    output_file = os.path.join(data_dir, "hgcal_electron_data_large_processed.pt")

    compute_and_save_features_vectorized(input_file, output_file, n_layers=28, batch_size=10000)
    print("Feature computation and saving completed.")