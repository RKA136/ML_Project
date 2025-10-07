import numpy as np
import cupy as cp
import torch
import h5py
from tqdm import tqdm
import json
import os

def load_h5(filename):
    """Load dataset from HDF5 file based on config.json data_dir."""
    dataset = {}
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    filepath = os.path.join(data_dir, filename)
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            dataset[key] = f[key][:]
    return dataset

def prepare_event_feature_tensors_gpu(filename, batch_size=20000):
    """
    Fully GPU-accelerated feature computation for calorimeter events.
    Computes per-event features including:
        E_sum, E_max, r_std, z_std, r90, and per-layer energy fractions.
    """
    data = load_h5(filename)
    nhits = data["nhits"].astype(np.int32)
    x_all, y_all, z_all, E_all = data["rechit_x"], data["rechit_y"], data["rechit_z"], data["rechit_energy"]
    true_E_all = data["target"].astype(np.float32)

    n_events = len(nhits)
    unique_zs = np.sort(np.unique(z_all))
    n_layers = len(unique_zs)

    feature_list = []
    start_idx = 0

    for batch_start in tqdm(range(0, n_events, batch_size), desc="GPU feature batches"):
        batch_end = min(batch_start + batch_size, n_events)
        nhits_batch = nhits[batch_start:batch_end]
        total_hits = nhits_batch.sum()

        # Slice hits for this batch
        batch_slice = slice(start_idx, start_idx + total_hits)
        x = cp.asarray(x_all[batch_slice], dtype=cp.float32)
        y = cp.asarray(y_all[batch_slice], dtype=cp.float32)
        z = cp.asarray(z_all[batch_slice], dtype=cp.float32)
        E = cp.asarray(E_all[batch_slice], dtype=cp.float32)

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
        x_cog = cp.bincount(event_ids, weights=x*E, minlength=batch_end - batch_start)/E_sum
        y_cog = cp.bincount(event_ids, weights=y*E, minlength=batch_end - batch_start)/E_sum
        z_cog = cp.bincount(event_ids, weights=z*E, minlength=batch_end - batch_start)/E_sum

        # Broadcast COG to hits
        x_cog_hits = x_cog[event_ids]
        y_cog_hits = y_cog[event_ids]
        z_cog_hits = z_cog[event_ids]

        # r and z residuals
        r = cp.sqrt((x - x_cog_hits)**2 + (y - y_cog_hits)**2)
        z_shift = z - z_cog_hits

        # Weighted std
        def weighted_std(vals):
            mean_sq = cp.bincount(event_ids, weights=vals**2 * E, minlength=batch_end - batch_start)/E_sum
            mean_val = cp.bincount(event_ids, weights=vals * E, minlength=batch_end - batch_start)/E_sum
            return cp.sqrt(cp.maximum(mean_sq - mean_val**2, 0))

        r_std = weighted_std(r)
        z_std = weighted_std(z_shift)

        # ------------------------
        # r90 calculation (per-event loop)
        # ------------------------
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
                r90[i] = r_sorted[min(idx90, r_sorted.size-1)]

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

    # Concatenate all batches
    X = np.vstack(feature_list)
    y = true_E_all[:X.shape[0]].reshape(-1, 1)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    print(f"Generated {X.shape[0]} events, {X.shape[1]} features each.")
    return X_tensor, y_tensor
