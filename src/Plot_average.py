# plot_average_layer_data.py

import os
import json
from EleReg import HGCALDataset, prepare_event_layer_dataframe, plot_average_energy_per_layer, plot_average_energy_per_layer_summary
import time

# ---------------------------
# 1. Configuration
# ---------------------------

filename = "hgcal_electron_data_0001.h5"
use_gpu = True   # Set True if you want GPU processing (requires CuPy)
batch_size = 100000  # GPU batch size (ignored if use_gpu=False)

# ---------------------------
# 2. Prepare Event-Layer DataFrame
# ---------------------------
time_start = time.time()
print("Preparing event-layer DataFrame...")
df = prepare_event_layer_dataframe(filename=filename, use_gpu=use_gpu, batch_size=batch_size)
print("DataFrame ready. Shape:", df.shape)
time_end = time.time()
print(f"Time taken to prepare the dataframe for use_gpu={use_gpu}:", time_end - time_start)

# ---------------------------
# 3. Plot Average Energy per Layer
# ---------------------------

plot_average_energy_per_layer(df)
print("Plot saved in figures directory.")

# ---------------------------
# 4. Plot Summary of Average Energy per Layer
# ---------------------------
# plot_average_energy_per_layer_summary(df, save_name= "average_energy_per_layer_summary_gpu.png" if use_gpu else "average_energy_per_layer_summary_cpu.png")
# print("Summary plot saved in figures directory.")