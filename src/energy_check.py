import json
import os
import h5py
import numpy as np

filename = "hgcal_electron_data_large.h5"

# Load configuration and construct full path
with open("config.json", "r") as f:
    config = json.load(f)
    data_dir = config["data_dir"]

filepath = os.path.join(data_dir, filename)

threshold = 0.5
chunk_size = 1_000_000  # adjust as per your RAM
count_total = 0
count_bad = 0

with h5py.File(filepath, "r") as f:
    # Access dataset without loading it
    E_dataset = f["rechit_energy"]
    n = len(E_dataset)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        E_chunk = E_dataset[start:end]  # loads only a slice
        count_total += len(E_chunk)
        count_bad += np.count_nonzero(E_chunk < threshold)

print("Total rechits:", count_total)
print("Bad rechits (E < 0.5 MIP):", count_bad)
print("Fraction bad:", count_bad / count_total)
