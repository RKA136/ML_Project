from Electron_Reg import prepare_event_layer_dataframe_cpu, plot_average_energy_per_layer
import time
# Prepare z-energy DataFrame
start_time = time.time()
z_energy_df = prepare_event_layer_dataframe_cpu(filename="hgcal_electron_data_0001.h5")
end_time = time.time()
print(z_energy_df.head())
print(z_energy_df.info())
print(f"DataFrame (CPU) prepared in {end_time - start_time:.6f} seconds")

# Plot average energy per layer
plot_average_energy_per_layer(z_energy_df)
