# ML_Project
Reconstruction of electrons with the CMS HGCAL beam-test prototype.

- For the data and figures folder path please create a `config.json` file and add the path to the data and the figures folder in the format 
```text
{
  "data_dir": "E:/GitHub/ML_Project/data",
  "figures_dir": "E:/GitHub/ML_Project/figures"
}
```

This will allow us to use the same code to get the data and save the figures in the same folder as we work.

The import will work as below
```python
import json
import os
# Load config.json
with open("config.json", "r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
figures_dir = config["figures_dir"]

dataset_path = os.path.join(data_dir, "hgcal_electron_data_0001.h5")
```
