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
    
    with load_h5(filename) as f:
        nhits = f["nhits"][:].astype(np.int32)
        n_events = len(nhits)
        
        unique_zs = np.sort(np.unique(f["rechit_z"][:]))
        n_layers = len(unique_zs)    
        
        feature_list = []
        start_index = 0
        
        for batch_start in tqdm(range(0,n_events,batch_size), desc = "Batches Processes"):
            batch_end = min(batch_start+batch_size,n_events)
            nhits_batch = nhits[batch_start:batch_end]
            total_hits = nhits_batch.sum()
            
            batch_slice = slice(start_index,start_index+total_hits)
            
            x = np.asarray(f["rechit_x"][batch_slice], dtype=np.float32)
            y = np.asarray(f["rechit_y"][batch_slice], dtype=np.float32)
            z = np.asarray(f["rechit_z"][batch_slice], dtype=np.float32)
            Energy = np.asarray(f["rechit_energy"][batch_slice], dtype=np.float32)
            
            z_bins = np.asarray(unique_zs, dtype=np.float32)
            layer_index = np.clip(np.searchsorted(z_bins,z), 0, n_layers - 1 )
            
            nhits_batch_gpu = np.asarray(nhits_batch, dtype=np.int32)
            cum_hits = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(nhits_batch_gpu)])
            hit_indices = np.arange(total_hits, dtype=np.int32)
            event_ids = np.searchsorted(cum_hits[1:], hit_indices, side='right')
            
            E_sum = np.bincount(event_ids, weights=Energy, minlength=batch_end - batch_start)
            E_sum = np.maximum(E_sum, 1e-8)

            
            # Energy per layer fraction
            linear_idx = event_ids * n_layers + layer_index
            E_layer_sum = np.bincount(linear_idx, weights=Energy, minlength=(batch_end - batch_start) * n_layers)
            E_layer_sum = E_layer_sum.reshape(batch_end - batch_start, n_layers)
            # E_layer_frac = E_layer_sum / E_sum[:, None]
            
            feats = np.concatenate([E_layer_sum], axis=1)
            
            feature_list.append(feats)
            
            start_index += total_hits
        
        true_E_all = f["true_energy"][:].astype(np.float32)
        
    X = np.vstack(feature_list)
    y = true_E_all[:X.shape[0]].reshape(-1,1)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    with open("config.json", "r") as f:
        config = json.load(f)
        data_dir = config["data_dir"]
    torch.save({"X": X_tensor, "y": y_tensor}, os.path.join(data_dir, "processed_data.pt"))
    print(f"Generated {X.shape[0]} events, {X.shape[1]} features each.")
    return X_tensor, y_tensor

if __name__ == "__main__":
    filename = "hgcal_electron_data_large.h5"
    per_event_feature_calculation(filename, 20000)