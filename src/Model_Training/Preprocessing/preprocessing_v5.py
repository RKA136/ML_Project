#!/usr/bin/env python3
"""
preprocessing_v5.py
-----------------------------------
This script extends the calorimeter preprocessing pipeline by replacing
the **layer-wise energy sum (E_sum)** with **layer-wise hit counts (N_hits)**.

Core Objective:
---------------
For each (event, layer), extract:
    - N_hits : number of hits in that layer
    - E1/E7  : core-to-near-ring ratio
    - E7/E19 : inner-to-outer-ring ratio

This feature set provides both geometric density and spatial energy distribution
information useful for machine-learning applications.

Pipeline Summary:
-----------------
1. **Lazy HDF5 Batch Streaming**
   - Efficiently processes large datasets by streaming batches of events.
   - Reads only required slices of flattened hit data per batch.

2. **Per-Layer Grouping**
   - Maps hits to (event, layer) using vectorized integer keys.
   - Avoids Python loops for performance.

3. **Feature Computation**
   - N_hits = count of hits per layer per event
   - E1/E7 and E7/E19 = spatial energy ratios defined around max-energy hit

4. **Incremental Storage**
   - Saves all results to a `np.memmap` array on disk.
   - Outputs final `.npy` and `.pt` formats.

Example:
--------
    python preprocessing_v5.py

Outputs:
---------
- `features_hits.npy` → NumPy memmap with extracted features
- `processed_data_hits_v5.pt` → PyTorch tensor dataset (X, y)
"""

import os
import json
import h5py
import numpy as np
from tqdm import tqdm
import torch

# Optional GPU acceleration
USE_CUPY = False
if USE_CUPY:
    import cupy as cp
    xp = cp
else:
    xp = np

# Geometrical constants for hexagonal cell ring definitions
HEX_SPACING = 1.1221
R1 = HEX_SPACING
R2A = np.sqrt(3.0) * HEX_SPACING
R2B = 2.0 * HEX_SPACING
R_TOL = 0.05
Z_TOL = 1e-3

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def open_h5(filename):
    cfg = load_config()
    path = os.path.join(cfg["data_dir"], filename)
    return h5py.File(path, "r")

def prepare_memmap(out_path, n_events, n_features, dtype=np.float32):
    """Create or open an np.memmap to store features incrementally."""
    shape = (n_events, n_features)
    mmap = np.memmap(out_path, dtype=dtype, mode="w+", shape=shape)
    return mmap

def process_batches(filename, batch_size=20000, use_cupy=False, out_filename="features_hits.npy"):
    """
    Batchwise fully vectorized processing.

    Computes per-layer:
        * N_hits (number of hits per layer per event)
        * E1/E7 ratio
        * E7/E19 ratio
    """
    if use_cupy and not xp is cp:
        import cupy as cp_local

    cfg = load_config()
    data_dir = cfg["data_dir"]
    fullpath = os.path.join(data_dir, filename)

    with h5py.File(fullpath, "r") as f:
        nhits_all = f["nhits"][:].astype(np.int64)
        n_events = len(nhits_all)
        unique_zs = np.sort(np.unique(f["rechit_z"][:]))
        L = len(unique_zs)

        # Prepare memmap for features
        n_features = 3 * L
        out_path = os.path.join(data_dir, out_filename)
        feature_mmap = prepare_memmap(out_path, n_events, n_features, dtype=np.float32)

        # Precompute cumulative hit indices
        nhits_cum = np.concatenate([[0], np.cumsum(nhits_all)])

        for batch_start in tqdm(range(0, n_events, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, n_events)
            B = batch_end - batch_start

            start_hit = int(nhits_cum[batch_start])
            end_hit = int(nhits_cum[batch_end])
            total_hits = end_hit - start_hit

            if total_hits == 0:
                feature_mmap[batch_start:batch_end, :] = 0.0
                continue

            # Load hit-level data
            x_flat = np.asarray(f["rechit_x"][start_hit:end_hit], dtype=np.float32)
            y_flat = np.asarray(f["rechit_y"][start_hit:end_hit], dtype=np.float32)
            z_flat = np.asarray(f["rechit_z"][start_hit:end_hit], dtype=np.float32)
            E_flat = np.asarray(f["rechit_energy"][start_hit:end_hit], dtype=np.float32)

            nhits_batch = nhits_all[batch_start:batch_end]
            cum_batch = np.concatenate([[0], np.cumsum(nhits_batch)])
            hit_idx_local = np.arange(total_hits)
            event_ids = np.searchsorted(cum_batch[1:], hit_idx_local, side="right").astype(np.int32)

            # Compute layer index for each hit
            layer_index = np.searchsorted(unique_zs, z_flat)
            layer_index = np.clip(layer_index, 0, L - 1).astype(np.int32)

            # Group by (event, layer)
            keys = event_ids.astype(np.int64) * np.int64(L) + layer_index.astype(np.int64)
            sort_idx = np.argsort(keys, kind="stable")
            keys_s = keys[sort_idx]
            x_s, y_s, E_s = x_flat[sort_idx], y_flat[sort_idx], E_flat[sort_idx]
            eids_s = (keys_s // np.int64(L)).astype(np.int32)
            lids_s = (keys_s % np.int64(L)).astype(np.int32)

            # Find unique groups and max hits per group
            unique_keys, start_pos, counts = np.unique(keys_s, return_index=True, return_counts=True)
            H = int(counts.max()) if counts.size > 0 else 0
            if H == 0:
                feature_mmap[batch_start:batch_end, :] = 0.0
                continue

            # Construct padded hit tensors
            x_pad = np.zeros((B, L, H), dtype=np.float32)
            y_pad = np.zeros((B, L, H), dtype=np.float32)
            E_pad = np.zeros((B, L, H), dtype=np.float32)
            mask = np.zeros((B, L, H), dtype=bool)

            pos = np.arange(len(keys_s)) - np.repeat(start_pos, counts)
            x_pad[eids_s, lids_s, pos] = x_s
            y_pad[eids_s, lids_s, pos] = y_s
            E_pad[eids_s, lids_s, pos] = E_s
            mask[eids_s, lids_s, pos] = True

            # --- Compute layer-wise hit counts ---
            N_hits = mask.sum(axis=2).astype(np.float32)

            # --- Energy ratio computations (same as v4) ---
            E_masked = np.where(mask, E_pad, 0.0)
            argmax = E_masked.argmax(axis=2)
            E_max = np.take_along_axis(E_masked, argmax[..., None], axis=2).squeeze(-1)
            x_max = np.take_along_axis(x_pad, argmax[..., None], axis=2)
            y_max = np.take_along_axis(y_pad, argmax[..., None], axis=2)

            dx = x_pad - x_max
            dy = y_pad - y_max
            dist = np.hypot(dx, dy)
            dist = np.where(mask, dist, np.inf)

            ring1 = (np.abs(dist - R1) <= R_TOL)
            ring2 = (np.abs(dist - R2A) <= R_TOL) | (np.abs(dist - R2B) <= R_TOL)
            E_ring1 = (E_masked * ring1).sum(axis=2)
            E_ring2 = (E_masked * ring2).sum(axis=2)
            E7 = E_max + E_ring1
            E19 = E7 + E_ring2

            E1_over_E7 = np.divide(E_max, E7, out=np.zeros_like(E7), where=E7 > 0)
            E7_over_E19 = np.divide(E7, E19, out=np.zeros_like(E19), where=E19 > 0)

            feats = np.concatenate([N_hits, E1_over_E7, E7_over_E19], axis=1)
            feature_mmap[batch_start:batch_end, :] = feats.astype(np.float32)
            feature_mmap.flush()

        # Save target values and final torch tensor
        true_energy = f["true_energy"][:n_events].astype(np.float32)

    out_torch = os.path.join(data_dir, "processed_data_large_v5.pt")
    torch.save({"X": torch.from_numpy(np.asarray(feature_mmap)),
                "y": torch.from_numpy(true_energy.reshape(-1, 1))}, out_torch)

    print(f"Finished. Features saved to: {out_path}")
    print(f"Torch dataset saved to: {out_torch}")
    return out_path, out_torch


if __name__ == "__main__":
    cfg = load_config()
    data_dir = cfg["data_dir"]
    filename = "hgcal_electron_data_large.h5"
    process_batches(filename, batch_size=5000, use_cupy=False, out_filename="features.npy")
