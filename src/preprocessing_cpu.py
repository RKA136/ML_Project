import torch
import h5py
import json
import os
from tqdm import tqdm

def prepare_event_feature_tensors_cpu_lazy(filename, batch_size=20000):
    """
    CPU-based, lazy-loading feature computation for calorimeter events.
    Computes per-event features including:
        E_sum, E_max, r_std, z_std, r90, and per-layer energy fractions.
    Uses HDF5 lazy access (loads only required slices per batch).
    """
    # ---------------------------
    # Load config and open HDF5
    # ---------------------------
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    filepath = os.path.join(data_dir, filename)

    with h5py.File(filepath, "r") as f:
        nhits_all = torch.tensor(f["nhits"][:], dtype=torch.int32)
        z_all = torch.tensor(f["rechit_z"][:], dtype=torch.float32)  # used for layer bins
        unique_zs = torch.sort(torch.unique(z_all)).values
        n_layers = len(unique_zs)
        true_E_all = torch.tensor(f["target"][:], dtype=torch.float32)
        n_events = len(nhits_all)

        # Precompute hit offsets to access hits lazily
        cum_hits = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(nhits_all.to(torch.int64), dim=0)])
        total_hits = int(cum_hits[-1])

        feature_list = []

        # ---------------------------
        # Process batches
        # ---------------------------
        for batch_start in tqdm(range(0, n_events, batch_size), desc="Lazy CPU feature batches"):
            batch_end = min(batch_start + batch_size, n_events)
            nhits_batch = nhits_all[batch_start:batch_end]
            total_hits_batch = int(nhits_batch.sum())

            # Compute hit range for this batch
            start_idx = int(cum_hits[batch_start])
            end_idx = int(cum_hits[batch_end])
            assert total_hits_batch == (end_idx - start_idx)

            # Lazy load just this slice
            x = torch.tensor(f["rechit_x"][start_idx:end_idx], dtype=torch.float32)
            y = torch.tensor(f["rechit_y"][start_idx:end_idx], dtype=torch.float32)
            z = torch.tensor(f["rechit_z"][start_idx:end_idx], dtype=torch.float32)
            E = torch.tensor(f["rechit_energy"][start_idx:end_idx], dtype=torch.float32)

            # Layer index
            layer_idx = torch.bucketize(z, unique_zs) - 1
            layer_idx = torch.clamp(layer_idx, 0, n_layers - 1)

            # Event IDs
            local_cum_hits = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(nhits_batch, dim=0)])
            hit_indices = torch.arange(total_hits_batch, dtype=torch.int32)
            event_ids = torch.searchsorted(local_cum_hits[1:], hit_indices)

            # Total energy
            E_sum = torch.zeros(batch_end - batch_start, dtype=torch.float32)
            E_sum.index_add_(0, event_ids, E)
            E_sum = torch.clamp(E_sum, min=1e-8)

            # Energy-weighted COG
            x_cog = torch.zeros_like(E_sum)
            y_cog = torch.zeros_like(E_sum)
            z_cog = torch.zeros_like(E_sum)
            x_cog.index_add_(0, event_ids, x * E)
            y_cog.index_add_(0, event_ids, y * E)
            z_cog.index_add_(0, event_ids, z * E)
            x_cog /= E_sum
            y_cog /= E_sum
            z_cog /= E_sum

            # Broadcast COG to hits
            x_cog_hits = x_cog[event_ids]
            y_cog_hits = y_cog[event_ids]
            z_cog_hits = z_cog[event_ids]

            # Residuals
            r = torch.sqrt((x - x_cog_hits) ** 2 + (y - y_cog_hits) ** 2)
            z_shift = z - z_cog_hits

            # Weighted std
            def weighted_std(vals):
                mean_sq = torch.zeros_like(E_sum)
                mean_val = torch.zeros_like(E_sum)
                mean_sq.index_add_(0, event_ids, (vals ** 2) * E)
                mean_val.index_add_(0, event_ids, vals * E)
                mean_sq /= E_sum
                mean_val /= E_sum
                return torch.sqrt(torch.clamp(mean_sq - mean_val ** 2, min=0.0))

            r_std = weighted_std(r)
            z_std = weighted_std(z_shift)

            # r90 per event
            r90 = torch.zeros(batch_end - batch_start, dtype=torch.float32)
            for i in range(batch_end - batch_start):
                mask = event_ids == i
                if mask.any():
                    r_ev = r[mask]
                    E_ev = E[mask]
                    order = torch.argsort(r_ev)
                    r_sorted = r_ev[order]
                    E_sorted = E_ev[order]
                    cumE = torch.cumsum(E_sorted, dim=0)
                    idx90 = torch.searchsorted(cumE, 0.9 * cumE[-1])
                    r90[i] = r_sorted[min(idx90.item(), len(r_sorted) - 1)]

            # E_max per event
            E_max = torch.zeros(batch_end - batch_start, dtype=torch.float32)
            for i in range(batch_end - batch_start):
                mask = event_ids == i
                if mask.any():
                    E_max[i] = E[mask].max()

            # Energy fraction per layer
            linear_idx = event_ids * n_layers + layer_idx
            E_layer_sum = torch.zeros((batch_end - batch_start) * n_layers, dtype=torch.float32)
            E_layer_sum.index_add_(0, linear_idx, E)
            E_layer_sum = E_layer_sum.view(batch_end - batch_start, n_layers)
            E_layer_frac = E_layer_sum / E_sum[:, None]

            # Combine features
            feats = torch.cat([
                E_sum[:, None],
                E_max[:, None],
                r_std[:, None],
                z_std[:, None],
                r90[:, None],
                E_layer_frac
            ], dim=1)

            feature_list.append(feats)

        # Concatenate results across all batches
        X = torch.vstack(feature_list)
        y = true_E_all[:X.shape[0]].view(-1, 1)

        # Save results
        torch.save({"X": X, "y": y}, os.path.join(data_dir, "processed_data_cpu.pt"))
        print(f"Generated {X.shape[0]} events, {X.shape[1]} features each.")
        return X, y