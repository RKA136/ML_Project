#!/usr/bin/env python3
"""
preprocessing_v1_reordered.py
-----------------------------------
Modified from `preprocessing_v1.py`:
Places the per-layer energy fractions (`E_layer_frac`) as the **first 28 columns**
of the output feature matrix `X`. The remaining global features are appended after.

Final feature order:
    [E_layer_frac (28 cols), E_sum, E_max, r_std, z_std]
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
    f = h5py.File(filepath, "r")
    return f

def prepare_event_feature_tensors_gpu(filename, batch_size=20000):
    """Fully GPU-accelerated feature computation for calorimeter events."""
    with load_h5(filename) as f:
        nhits = f["nhits"][:].astype(np.int32)
        n_events = len(nhits)

        unique_zs = np.sort(np.unique(f["rechit_z"][:]))
        n_layers = len(unique_zs)

        feature_list = []
        start_idx = 0

        for batch_start in tqdm(range(0, n_events, batch_size), desc="GPU feature batches"):
            batch_end = min(batch_start + batch_size, n_events)
            nhits_batch = nhits[batch_start:batch_end]
            total_hits = nhits_batch.sum()
            batch_slice = slice(start_idx, start_idx + total_hits)

            # GPU arrays
            x = cp.asarray(f["rechit_x"][batch_slice], dtype=cp.float32)
            y = cp.asarray(f["rechit_y"][batch_slice], dtype=cp.float32)
            z = cp.asarray(f["rechit_z"][batch_slice], dtype=cp.float32)
            E = cp.asarray(f["rechit_energy"][batch_slice], dtype=cp.float32)

            # Layer indices
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

            # Weighted COG
            x_cog = cp.bincount(event_ids, weights=x * E, minlength=batch_end - batch_start) / E_sum
            y_cog = cp.bincount(event_ids, weights=y * E, minlength=batch_end - batch_start) / E_sum
            z_cog = cp.bincount(event_ids, weights=z * E, minlength=batch_end - batch_start) / E_sum

            x_cog_hits = x_cog[event_ids]
            y_cog_hits = y_cog[event_ids]
            z_cog_hits = z_cog[event_ids]

            r = cp.sqrt((x - x_cog_hits) ** 2 + (y - y_cog_hits) ** 2)
            z_shift = z - z_cog_hits

            # Weighted std function
            def weighted_std(vals):
                mean_sq = cp.bincount(event_ids, weights=vals**2 * E, minlength=batch_end - batch_start) / E_sum
                mean_val = cp.bincount(event_ids, weights=vals * E, minlength=batch_end - batch_start) / E_sum
                return cp.sqrt(cp.maximum(mean_sq - mean_val**2, 0))

            r_std = weighted_std(r)
            z_std = weighted_std(z_shift)

            # Max hit energy per event
            E_max = cp.zeros(batch_end - batch_start, dtype=cp.float32)
            cp.maximum.at(E_max, event_ids, E)

            # Energy per layer fraction
            linear_idx = event_ids * n_layers + layer_idx
            E_layer_sum = cp.bincount(linear_idx, weights=E, minlength=(batch_end - batch_start) * n_layers)
            E_layer_sum = E_layer_sum.reshape(batch_end - batch_start, n_layers)
            E_layer_frac = E_layer_sum / E_sum[:, None]

            # Concatenate with E_layer_frac first
            feats = cp.concatenate([
                E_layer_frac,                   # columns 0–27
                E_sum[:, None],                 # column 28
                E_max[:, None],                 # column 29
                r_std[:, None],                 # column 30
                z_std[:, None]                  # column 31
            ], axis=1)

            # Move to CPU
            feature_list.append(cp.asnumpy(feats))
            start_idx += total_hits
            cp.get_default_memory_pool().free_all_blocks()

        true_E_all = f["target"][:].astype(np.float32)

    # Combine batches
    X = np.vstack(feature_list)
    y = true_E_all[:X.shape[0]].reshape(-1, 1)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Save tensors
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    out_path = os.path.join(data_dir, "processed_data_0001_v1.pt")
    torch.save({"X": X_tensor, "y": y_tensor}, out_path)

    print(f"Generated {X.shape[0]} events, {X.shape[1]} features each.")
    print(f"Saved reordered feature tensor to {out_path}")

    return X_tensor, y_tensor

if __name__ == "__main__":
    filename = "hgcal_electron_data_0001.h5"
    prepare_event_feature_tensors_gpu(filename, 20000)
