#!/usr/bin/env python3
"""
preprocessing_v3.py
-----------------------------------
This script performs **batch-wise per-event feature computation** from raw HDF5
high-granularity calorimeter (HGCAL) simulation data. It efficiently processes
large datasets without exhausting memory by reading only the required slices at a time.

Objective:
----------
Compute **layer-wise energy depositions** for each event and store them in
a structured tensor dataset for later use in regression or classification models.

Overview of Processing Steps:
-----------------------------
1. **Lazy HDF5 Loading**
   - The file is opened using a lightweight handler via `load_h5()`,
     ensuring large datasets are processed efficiently without full memory loading.

2. **Event and Layer Mapping**
   - Events are identified using the cumulative `nhits` array.
   - The detector's layers are determined from the unique `rechit_z` values.
   - Each hit is assigned to both an event index and a layer index.

3. **Feature Construction**
   - For each batch of events:
       - Energy per hit is accumulated per layer using `np.bincount`.
       - The resulting 2D matrix has shape `(n_events_batch, n_layers)`,
         where each row represents a single event’s per-layer energy deposition.

4. **Data Assembly and Output**
   - Batches are concatenated into a full dataset.
   - The dataset is saved as a PyTorch `.pt` file containing:
         {"X": X_tensor, "y": y_tensor}
     where:
       - X_tensor → layer-wise energy sums
       - y_tensor → true total event energy (target variable)

5. **Advantages**
   - Memory-safe for large-scale data.
   - Fully vectorized numpy operations ensure computational efficiency.
   - Compatible with both CPU and GPU-based later stages.

Output:
-------
- `processed_data.pt` stored in the configured data directory
   (defined in `config.json`).

Example:
--------
    python per_event_feature_calculation.py
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

def per_event_feature_calculation(filename, batch_size=20000):
    """
    Compute per-event calorimeter features (energy per layer) in batches.

    Args:
        filename (str): Name of the input HDF5 dataset file.
        batch_size (int): Number of events to process per batch.

    Returns:
        X_tensor (torch.Tensor): Feature tensor of shape (n_events, n_layers).
        y_tensor (torch.Tensor): True energy tensor of shape (n_events, 1).
    """
    with load_h5(filename) as f:
        nhits = f["nhits"][:].astype(np.int32)
        n_events = len(nhits)
        
        # Identify calorimeter layers
        unique_zs = np.sort(np.unique(f["rechit_z"][:]))
        n_layers = len(unique_zs)    
        
        feature_list = []
        start_index = 0
        
        for batch_start in tqdm(range(0, n_events, batch_size), desc="Batches Processed"):
            batch_end = min(batch_start + batch_size, n_events)
            nhits_batch = nhits[batch_start:batch_end]
            total_hits = nhits_batch.sum()
            
            # Load only relevant hits
            batch_slice = slice(start_index, start_index + total_hits)
            x = np.asarray(f["rechit_x"][batch_slice], dtype=np.float32)
            y = np.asarray(f["rechit_y"][batch_slice], dtype=np.float32)
            z = np.asarray(f["rechit_z"][batch_slice], dtype=np.float32)
            Energy = np.asarray(f["rechit_energy"][batch_slice], dtype=np.float32)
            
            # Map hits to layer indices
            z_bins = np.asarray(unique_zs, dtype=np.float32)
            layer_index = np.clip(np.searchsorted(z_bins, z), 0, n_layers - 1)
            
            # Assign event IDs to hits
            nhits_batch_gpu = np.asarray(nhits_batch, dtype=np.int32)
            cum_hits = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(nhits_batch_gpu)])
            hit_indices = np.arange(total_hits, dtype=np.int32)
            event_ids = np.searchsorted(cum_hits[1:], hit_indices, side='right')
            
            # Compute total deposited energy per event
            E_sum = np.bincount(event_ids, weights=Energy, minlength=batch_end - batch_start)
            E_sum = np.maximum(E_sum, 1e-8)

            # Compute layer-wise energy sums
            linear_idx = event_ids * n_layers + layer_index
            E_layer_sum = np.bincount(linear_idx, weights=Energy, minlength=(batch_end - batch_start) * n_layers)
            E_layer_sum = E_layer_sum.reshape(batch_end - batch_start, n_layers)
            
            # Concatenate (only per-layer energies used)
            feats = np.concatenate([E_layer_sum], axis=1)
            feature_list.append(feats)
            
            start_index += total_hits
        
        # Load true energies
        true_E_all = f["target"][:].astype(np.float32)
        
    # Stack all batches
    X = np.vstack(feature_list)
    y = true_E_all[:X.shape[0]].reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Save to configured directory
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    torch.save({"X": X_tensor, "y": y_tensor}, os.path.join(data_dir, "processed_data_0001_v3.pt"))
    
    print(f"Generated {X.shape[0]} events, {X.shape[1]} features each.")
    return X_tensor, y_tensor

if __name__ == "__main__":
    filename = "hgcal_electron_data_0001.h5"
    per_event_feature_calculation(filename, 20000)