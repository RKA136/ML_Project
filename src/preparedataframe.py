from enereg import prepare_event_feature_tensors_gpu
import time 

filename = "hgcal_electron_data_0001.h5"

start_time = time.time()
X_tensor, y_tensor = prepare_event_feature_tensors_gpu(filename, batch_size=20000)
end_time = time.time()

print(f"Time taken to prepare event feature tensors: {end_time - start_time} seconds")