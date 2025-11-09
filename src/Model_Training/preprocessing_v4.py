import os
import json
import h5py
import numpy as np
from tqdm import tqdm
import torch

# If you want GPU, set USE_CUPY = True and have cupy installed & GPU available.
USE_CUPY = False
if USE_CUPY:
    import cupy as cp
    xp = cp
else:
    xp = np

# Constants you can tune
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
    # file size = n_events * n_features * dtype.itemsize
    shape = (n_events, n_features)
    mmap = np.memmap(out_path, dtype=dtype, mode="w+", shape=shape)
    return mmap

def process_batch_vectorized(f, nhits_batch, batch_start, batch_end, unique_zs, xp=np):
    """
    Vectorized per-batch processing. Returns features array shape (B, 3*L)
    using xp (np or cupy).
    No python loops over events/layers/hits inside heavy math; only batch loop is external.
    """
    B = batch_end - batch_start
    L = len(unique_zs)

    total_hits = nhits_batch.sum()
    if total_hits == 0:
        return np.zeros((B, 3 * L), dtype=np.float32)

    # Compute per-hit event_ids
    cum = np.concatenate([[0], np.cumsum(nhits_batch)])
    # event id for each hit in batch (0..B-1)
    event_ids = np.searchsorted(cum[1:], np.arange(total_hits), side="right")

    # read hit arrays
    # Determine slice indices in the file: we must know absolute start: compute global start index of this batch
    # We assume the caller provides the file slice correctly: f[...] reading is done using an absolute offset
    # For this function we will expect the caller passes already sliced arrays.
    # But to keep API simple, we'll compute full-file start index externally.
    raise RuntimeError("Internal function; use process_batches() wrapper below.")

def process_batches(filename, batch_size=20000, use_cupy=False, out_filename="features.npy"):
    """
    Full streaming + fully-vectorized per-batch pipeline.
    - Reads ragged HDF5 format (nhits, flattened rechit_* arrays, true_energy)
    - For each batch, builds padded tensors WITHOUT nested Python loops by:
        * sorting hits by (event_id, layer_index)
        * computing pos_in_group vectorized
        * vectorized scatter into x_pad/y_pad/E_pad
    - Computes E_sum, E1/E7, E7/E19 with broadcasting
    - Appends results into an np.memmap (out_filename)
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

        # prepare memmap to hold features (3 per layer)
        n_features = 3 * L
        out_path = os.path.join(data_dir, out_filename)
        feature_mmap = prepare_memmap(out_path, n_events, n_features, dtype=np.float32)

        # We'll need to track the absolute start index into flattened rechit arrays
        # Precompute prefix sums of nhits to index into flattened arrays
        nhits_cum = np.concatenate([[0], np.cumsum(nhits_all)])

        for batch_start in tqdm(range(0, n_events, batch_size), desc="batches"):
            batch_end = min(batch_start + batch_size, n_events)
            B = batch_end - batch_start

            # compute hit slice for this batch in flattened arrays
            start_hit = int(nhits_cum[batch_start])
            end_hit = int(nhits_cum[batch_end])
            total_hits = end_hit - start_hit
            if total_hits == 0:
                feature_mmap[batch_start:batch_end, :] = 0.0
                continue

            # load hit-level arrays for this batch
            x_flat = np.asarray(f["rechit_x"][start_hit:end_hit], dtype=np.float32)
            y_flat = np.asarray(f["rechit_y"][start_hit:end_hit], dtype=np.float32)
            z_flat = np.asarray(f["rechit_z"][start_hit:end_hit], dtype=np.float32)
            E_flat = np.asarray(f["rechit_energy"][start_hit:end_hit], dtype=np.float32)

            # compute event_ids (0..B-1) for each hit in this batch
            # event id = arg such that start_hit + offset falls in event range:
            # we can compute by searching nhits cumulative restricted to this batch
            nhits_batch = nhits_all[batch_start:batch_end]
            cum_batch = np.concatenate([[0], np.cumsum(nhits_batch)])
            # event_ids for hits in this batch
            hit_idx_local = np.arange(total_hits)
            event_ids = np.searchsorted(cum_batch[1:], hit_idx_local, side="right").astype(np.int32)

            # compute layer index for each hit
            # find index in unique_zs such that unique_zs[layer_index] == z approximately
            # We assume z_flat values match unique_zs up to small rounding
            # Use np.searchsorted and then clamp
            layer_index = np.searchsorted(unique_zs, z_flat)
            layer_index = np.clip(layer_index, 0, L - 1).astype(np.int32)

            # Build grouping key = event_id * L + layer_index
            keys = event_ids.astype(np.int64) * np.int64(L) + layer_index.astype(np.int64)

            # Sort hits by key so groups are contiguous
            sort_idx = np.argsort(keys, kind="stable")
            keys_s = keys[sort_idx]
            x_s = x_flat[sort_idx]
            y_s = y_flat[sort_idx]
            E_s = E_flat[sort_idx]
            event_ids_s = event_ids[sort_idx].astype(np.int32)
            layer_idx_s = layer_index[sort_idx].astype(np.int32)

            # Unique keys and group starts/counts
            unique_keys, start_pos, counts = np.unique(keys_s, return_index=True, return_counts=True)
            # if no groups, create empty arrays
            # compute position within group for each element: pos = arange(len(keys_s)) - repeat(start_pos, counts)
            if len(keys_s) > 0:
                reps = np.repeat(start_pos, counts)
                positions = np.arange(len(keys_s), dtype=np.int32) - reps
            else:
                positions = np.array([], dtype=np.int32)

            # decode event, layer for each sorted element (vectorized)
            eids_s = (keys_s // np.int64(L)).astype(np.int32)
            lids_s = (keys_s % np.int64(L)).astype(np.int32)

            # Determine H = max hits per (event,layer) for this batch
            if counts.size > 0:
                H = int(counts.max())
            else:
                H = 0
            if H == 0:
                feature_mmap[batch_start:batch_end, :] = 0.0
                continue

            # Allocate padded arrays (NumPy). Use xp if cupy desired.
            x_pad = np.zeros((B, L, H), dtype=np.float32)
            y_pad = np.zeros((B, L, H), dtype=np.float32)
            E_pad = np.zeros((B, L, H), dtype=np.float32)
            mask = np.zeros((B, L, H), dtype=bool)

            # Vectorized scatter: convert eids_s, lids_s, positions to arrays for fancy indexing
            # Note: advanced indexing assignment supports vectorized assignment
            # But NumPy requires indices arrays of same shape; we already have them
            # Only assign where positions < H (they will be by construction)
            idx_e = eids_s
            idx_l = lids_s
            idx_p = positions

            # Perform scatter assignment (vectorized)
            x_pad[idx_e, idx_l, idx_p] = x_s
            y_pad[idx_e, idx_l, idx_p] = y_s
            E_pad[idx_e, idx_l, idx_p] = E_s
            mask[idx_e, idx_l, idx_p] = True

            # Optionally move to GPU (cupy) by xp.asarray
            if use_cupy:
                x_pad = cp.asarray(x_pad); y_pad = cp.asarray(y_pad)
                E_pad = cp.asarray(E_pad); mask = cp.asarray(mask)
                R1c = cp.float32(R1); R2Ac = cp.float32(R2A); R2Bc = cp.float32(R2B); tol = cp.float32(R_TOL)
                # compute with cupy
                E_masked = cp.where(mask, E_pad, 0.0)
                E_sum = E_masked.sum(axis=2)
                argmax = E_masked.argmax(axis=2)  # shape (B,L)
                E_max = cp.take_along_axis(E_masked, argmax[..., None], axis=2).squeeze(-1)
                x_max = cp.take_along_axis(x_pad, argmax[..., None], axis=2)
                y_max = cp.take_along_axis(y_pad, argmax[..., None], axis=2)
                dx = x_pad - x_max
                dy = y_pad - y_max
                dist = cp.hypot(dx, dy)
                dist = cp.where(mask, dist, cp.inf)
                ring1 = (cp.abs(dist - R1c) <= tol)
                ring2 = (cp.abs(dist - R2Ac) <= tol) | (cp.abs(dist - R2Bc) <= tol)
                E_ring1 = (E_masked * ring1).sum(axis=2)
                E_ring2 = (E_masked * ring2).sum(axis=2)
                E7 = E_max + E_ring1
                E19 = E7 + E_ring2
                E1_over_E7 = cp.divide(E_max, E7, out=cp.zeros_like(E7), where=E7 > 0)
                E7_over_E19 = cp.divide(E7, E19, out=cp.zeros_like(E19), where=E19 > 0)
                # bring back to cpu (numpy) for memmap writing
                E_sum = cp.asnumpy(E_sum)
                E1_over_E7 = cp.asnumpy(E1_over_E7)
                E7_over_E19 = cp.asnumpy(E7_over_E19)
            else:
                # All numpy: vectorized operations
                E_masked = np.where(mask, E_pad, 0.0)
                E_sum = E_masked.sum(axis=2)                         # (B, L)
                argmax = E_masked.argmax(axis=2)                     # (B, L), index along H
                # gather x_max, y_max, E_max
                E_max = np.take_along_axis(E_masked, argmax[..., None], axis=2).squeeze(-1)
                x_max = np.take_along_axis(x_pad, argmax[..., None], axis=2)
                y_max = np.take_along_axis(y_pad, argmax[..., None], axis=2)
                # compute distances broadcasted
                dx = x_pad - x_max                                       # (B, L, H)
                dy = y_pad - y_max
                dist = np.hypot(dx, dy)
                dist = np.where(mask, dist, np.inf)                     # invalid positions set to inf
                ring1 = (np.abs(dist - R1) <= R_TOL)
                ring2 = (np.abs(dist - R2A) <= R_TOL) | (np.abs(dist - R2B) <= R_TOL)
                E_ring1 = (E_masked * ring1).sum(axis=2)
                E_ring2 = (E_masked * ring2).sum(axis=2)
                E7 = E_max + E_ring1
                E19 = E7 + E_ring2
                E1_over_E7 = np.divide(E_max, E7, out=np.zeros_like(E7), where=E7 > 0)
                E7_over_E19 = np.divide(E7, E19, out=np.zeros_like(E19), where=E19 > 0)

            # Stack features: [E_sum, E1/E7, E7/E19] along last axis per layer
            feats = np.concatenate([E_sum, E1_over_E7, E7_over_E19], axis=1)  # shape (B, 3L)

            # write to memmap
            feature_mmap[batch_start:batch_end, :] = feats.astype(np.float32)
            feature_mmap.flush()

        # optionally save true_energy too in separate memmap or torch file
        # save as torch at the end
        true_energy = f["target"][:n_events].astype(np.float32)

    # write torch file (single write)
    out_torch = os.path.join(data_dir, "processed_data.pt")
    torch.save({"X": torch.from_numpy(np.asarray(feature_mmap)), "y": torch.from_numpy(true_energy.reshape(-1,1))}, out_torch)
    print("Finished. Features saved to:", out_path, "and torch file:", out_torch)
    return out_path, out_torch

if __name__ == "__main__":
    cfg = load_config()
    data_dir = cfg["data_dir"]
    filename = "hgcal_electron_data_0001.h5"
    # experiment: smaller batch for limited RAM, e.g. 2000 or 5000
    process_batches(filename, batch_size=5000, use_cupy=False, out_filename="features.npy")
