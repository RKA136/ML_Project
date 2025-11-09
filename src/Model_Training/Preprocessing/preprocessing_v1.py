#!/usr/bin/env python3
"""
preprocessing_v1.py
-----------------------------------
This script performs fully GPU-accelerated feature extraction from calorimeter
HDF5 event data and saves the resulting feature and target tensors in PyTorch format.

Overview:
1. **Lazy HDF5 Loading**
   - Uses the `load_h5()` helper to open large `.h5` files without loading them
     entirely into memory.
   - Reads only the required slices of hit data per batch.

2. **GPU-Accelerated Feature Computation**
   - Uses CuPy arrays for massive speedup in batch-level computation.
   - Features are computed per event across multiple detector layers.

3. **Computed Features**
   For each calorimeter event, the following physical features are calculated:
   - `E_sum`: total deposited energy.
   - `E_max`: maximum single-hit energy.
   - `r_std`: RMS spread in the radial direction.
   - `z_std`: RMS spread along the detector depth (longitudinal axis).
   - `r90`: radial distance containing 90% of total event energy.
   - `E_layer_frac`: normalized energy fraction deposited per layer.

4. **Batch-wise Processing**
   - Processes events in batches (default: 20,000) to avoid GPU memory overflow.
   - Uses cumulative indexing to correctly associate hits with events.

5. **Data Output**
   - Combines all batches into a single NumPy array, converts to Torch tensors.
   - Saves final tensors to `processed_data_v1.pt` in the directory specified
     under `"data_dir"` in `config.json`.

6. **Advantages**
   - Enables high-performance feature extraction for very large datasets.
   - Minimizes CPU–GPU data transfer overhead.
   - Avoids out-of-memory errors through controlled batching.

Output:
- `processed_data.pt`: a PyTorch dictionary with tensors:
      {"X": feature_tensor, "y": target_tensor}

Example Use:
    X_tensor, y_tensor = prepare_event_feature_tensors_gpu("hgcal_electron_data_large.h5")
"""

import numpy as np
import cupy as cp
import torch
import h5py
from tqdm import tqdm
import json
import os

def load_h5(filename):
    """Lazy-load HDF5 dataset (returns file handle instead of full arrays)."""
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    filepath = os.path.join(data_dir, filename)

    # Return file handle (caller must close)
    f = h5py.File(filepath, "r")
    return f  # not reading into memory

def prepare_event_feature_tensors_gpu(filename, batch_size=20000):
    """
    Fully GPU-accelerated feature computation for calorimeter events.
    Uses lazy loading (reads only necessary slices of data at a time).
    """
    with load_h5(filename) as f:
        nhits = f["nhits"][:].astype(np.int32)
        n_events = len(nhits)

        # Precompute z-bin info
        unique_zs = np.sort(np.unique(f["rechit_z"][:]))
        n_layers = len(unique_zs)

        feature_list = []
        start_idx = 0

        # Loop over event batches
        for batch_start in tqdm(range(0, n_events, batch_size), desc="GPU feature batches"):
            batch_end = min(batch_start + batch_size, n_events)
            nhits_batch = nhits[batch_start:batch_end]
            total_hits = nhits_batch.sum()

            # Compute slice indices
            batch_slice = slice(start_idx, start_idx + total_hits)

            # Lazily load hit data for this batch only
            x = cp.asarray(f["rechit_x"][batch_slice], dtype=cp.float32)
            y = cp.asarray(f["rechit_y"][batch_slice], dtype=cp.float32)
            z = cp.asarray(f["rechit_z"][batch_slice], dtype=cp.float32)
            E = cp.asarray(f["rechit_energy"][batch_slice], dtype=cp.float32)

            # Layer index
            z_bins = cp.asarray(unique_zs, dtype=cp.float32)
            layer_idx = cp.clip(cp.searchsorted(z_bins, z), 0, n_layers - 1)

            # Event IDs
            nhits_batch_gpu = cp.asarray(nhits_batch, dtype=cp.int32)
            cum_hits = cp.concatenate([cp.array([0], dtype=cp.int32), cp.cumsum(nhits_batch_gpu)])
            hit_indices = cp.arange(total_hits, dtype=cp.int32)
            event_ids = cp.searchsorted(cum_hits[1:], hit_indices, side='right')

            # Total energy
            E_sum = cp.bincount(event_ids, weights=E, minlength=batch_end - batch_start)
            E_sum = cp.maximum(E_sum, 1e-8)

            # Energy-weighted COG
            x_cog = cp.bincount(event_ids, weights=x * E, minlength=batch_end - batch_start) / E_sum
            y_cog = cp.bincount(event_ids, weights=y * E, minlength=batch_end - batch_start) / E_sum
            z_cog = cp.bincount(event_ids, weights=z * E, minlength=batch_end - batch_start) / E_sum

            # Broadcast COG to hits
            x_cog_hits = x_cog[event_ids]
            y_cog_hits = y_cog[event_ids]
            z_cog_hits = z_cog[event_ids]

            # r and z residuals
            r = cp.sqrt((x - x_cog_hits) ** 2 + (y - y_cog_hits) ** 2)
            z_shift = z - z_cog_hits

            # Weighted std
            def weighted_std(vals):
                mean_sq = cp.bincount(event_ids, weights=vals**2 * E, minlength=batch_end - batch_start) / E_sum
                mean_val = cp.bincount(event_ids, weights=vals * E, minlength=batch_end - batch_start) / E_sum
                return cp.sqrt(cp.maximum(mean_sq - mean_val**2, 0))

            r_std = weighted_std(r)
            z_std = weighted_std(z_shift)

            # r90 calculation
            r90 = cp.zeros(batch_end - batch_start, dtype=cp.float32)
            for i in range(batch_end - batch_start):
                mask = event_ids == i
                if cp.any(mask):
                    r_ev = r[mask]
                    E_ev = E[mask]
                    order = cp.argsort(r_ev)
                    r_sorted = r_ev[order]
                    E_sorted = E_ev[order]
                    cumE = cp.cumsum(E_sorted)
                    idx90 = cp.searchsorted(cumE, 0.9 * cumE[-1])
                    r90[i] = r_sorted[min(idx90, r_sorted.size - 1)]

            # Max hit energy per event
            E_max = cp.zeros(batch_end - batch_start, dtype=cp.float32)
            cp.maximum.at(E_max, event_ids, E)

            # Energy per layer fraction
            linear_idx = event_ids * n_layers + layer_idx
            E_layer_sum = cp.bincount(linear_idx, weights=E, minlength=(batch_end - batch_start) * n_layers)
            E_layer_sum = E_layer_sum.reshape(batch_end - batch_start, n_layers)
            E_layer_frac = E_layer_sum / E_sum[:, None]

            # Stack features
            feats = cp.concatenate([
                E_sum[:, None], E_max[:, None], r_std[:, None], z_std[:, None], r90[:, None], E_layer_frac
            ], axis=1)

            # Move to CPU
            feature_list.append(cp.asnumpy(feats))
            start_idx += total_hits
            cp.get_default_memory_pool().free_all_blocks()

        # True energy (load only once)
        true_E_all = f["true_energy"][:].astype(np.float32)

    # Concatenate all batches
    X = np.vstack(feature_list)
    y = true_E_all[:X.shape[0]].reshape(-1, 1)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Save tensors
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    torch.save({"X": X_tensor, "y": y_tensor}, os.path.join(data_dir, "processed_data_v1.pt"))
    print(f"Generated {X.shape[0]} events, {X.shape[1]} features each.")
    return X_tensor, y_tensor
