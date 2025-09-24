from Electron_Reg import prepare_event_layer_dataframe_gpu, plot_average_energy_per_layer
import time

# Prepare z-energy DataFrame (GPU)
start_time = time.time()
z_energy_df_gpu = prepare_event_layer_dataframe_gpu(filename="hgcal_electron_data_0001.h5")
print(z_energy_df_gpu.head())
print(z_energy_df_gpu.info())
end_time = time.time()
print(f"DataFrame (GPU) prepared in {end_time - start_time:.6f} seconds")

# Plot average energy per layer
plot_average_energy_per_layer(z_energy_df_gpu)