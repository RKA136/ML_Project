from Electron_Reg import display_event, hits_per_event, true_energy_distribution, get_layer_positions

# Visualize a specific event (you can change the event_index)  
display_event(event_index=3556, filename="hgcal_electron_data_0001.h5")

# Plot hits per event
hits_per_event(filename="hgcal_electron_data_0001.h5")

# Plot true energy distribution
true_energy_distribution(filename="hgcal_electron_data_0001.h5")

# Get layer positions
layer_positions = get_layer_positions(filename="hgcal_electron_data_0001.h5")
print("Unique Layer Positions (z):", layer_positions)
