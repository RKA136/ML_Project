# ML Project

# Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Functions](#functions)
    1. [Load Data](#load-data)
    2. [Event Display](#event-display)
    3. [Hits Per Event](#hits-per-event)
    4. [True Energy Distribution](#true-energy-distrinution)
    5. [Prepare Event Layer Dataframe (CPU version)](#prepare-event-layer-dataframe-cpu-version)
    6. [Prepare Event Layer Dataframe (GPU Version)](#prepare-event-layer-dataframe-gpu-version)
    7. [Plot Average Energy Per Layer](#plot-average-energy-per-layer)


# Introduction

In this project we have to do the **Reconstruction of electrons with the CMS HGCAL beam-test prototype**.

- The reference papers for this project are
    - [ELECTRON ENERGY REGRESSION IN THE CMS HIGH-GRANULARITY CALORIMETER PROTOTYPE](references/2309.06582v1.pdf)
    - [Response of a CMS HGCAL silicon-pad electromagnetic calorimeter prototype to 20-300 GeV positrons](references/2111.06855v3.pdf)

The struucture of the ML_Project folder is 
```text
ML_Project
|---/data/ # Contains the data files for the model training
|---/figures/ # Contains the figures and plots.
|---/references/ # Contains the reference papers for this project.
|---/src/ # Contains the codes used in this project
      |---Electron_Reg # Contains all the funtions used in the project, the functions are called in other .py files for cleaner presentation.
|---.gitignore # ignores the data and config files while pushing to the GitHub repository
|---config.json # gives the local location of data and figures folder
|---README.md
|---report.md #Contains the oveview of the whole project
```

# Dataset

As stated in the [ELECTRON ENERGY REGRESSION IN THE CMS HIGH-GRANULARITY CALORIMETER PROTOTYPE](references/2309.06582v1.pdf) paper the dataset consists of simulation of reconstructed hits known as **rechits**, produced by the positrons passing through the HGCAL test beam prototype. 
- The hits are chosen to have a minimum energy of 0.5 MIP, which is well above the HGCAL noise level.
- Events with more than 50 hits in CE-H layers are rejected.
- The track of electron extrapolated using the hits from the DWC chambers is requires to be within a 2×2 cm² window within the first layer.
- The final dataset is a set of *3.2 million* events.
- Each event contains position coordinates and calibrated energies of the rechits within the detector.
- HDF5 format is used to organize the data in hierarchical arrays.
- The array structure is 
    - `nhits` - An integer array representing number of reconstructed hits (rechits) in each event.
    - `rechit_x` - A nested array of length equal to the number of events and sub-arrays of length of nhits. Each subarray contains a floating value representing x-coordinate of the position of the rechits in units of centimeters.
    - `rechit_y` - A nested array with a structure and size same as rechit_x. Each floating value represents the y-coordinate of the position of a rechit in units of centimeters.
    - `rechit_z` - A nested array with a structure and size same as rechit_x. Each floating value represents the z-coordinate of the position of a rechit in units of centimeters.
    - `rechit_energy` - A nested array with a structure and size same as rechit_x. Each floating value represents the calibrated energy of a rechit in units of **MIPs**.
    - `target` - The true energy of the incoming positron in units of **GeV**.

- The dataset consists of two files in .h5 format. 
- There are two datasets 
    - 1. hgcal_electron_data_0001.h5 - Smaller dataset.
    - 2. hgcal_electron_data_large.h5 - Full dataset.

# Functions
Here we discuss about the function used in this project and explain the operations. All the functions are in the [Electron_Reg](src/Electron_Reg.py) file.

## Load Data

- `load_data` loads the HGCAL data from the .h5 file and converst it to a ready to use format. 

```python
def load_data(filename="hgcal_electron_data_0001.h5"):
    """Load the HGCAL electron dataset from an HDF5 file.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    Returns:
        dict: A dictionary containing the dataset.
    """
```

## Event Display

- `display_event` helps to open and show a 3d interactive plot of a electron hit event. 
- The dataset contains a number of events of the electron shower.
- The user can choose which event he/she wants to visualize and give it as an argument in the function.
- The function creates a .html file which can be opened in a browser and can be interacted to fully visualize the hit events and the energy deposited in the points.

```python
def display_event(event_index=200, filename="hgcal_electron_data_0001.h5"):
    """Display a 3D event from the HGCAL electron dataset.
    Args:
        event_index (int): Index of the event to display.
        filename (str): Name of the HDF5 file containing the dataset.
    Returns: html file with interactive 3D plot of the event.
    """
```

## Hits per Event

- `hits_per_event` plots the number of hits in each event as a histogram and saves the plot in the figures folder.

```python
def hits_per_event(filename="hgcal_electron_data_0001.h5"):
    """Plot a histogram of the number of hits per event in the dataset.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    """
```

## True Energy Distrinution

- `true_energy_distribution` plots the true energies in a histogram and saves the figure in the figures folder.

```python
def true_energy_distribution(filename="hgcal_electron_data_0001.h5"):
    """Plot a histogram of the true energy distribution of events in the dataset.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    """
```

## Prepare Event Layer Dataframe (CPU version)

### Description
Processes calorimeter hit data stored in an HDF5 file and prepares a **pandas DataFrame** containing the **average energy deposited per detector layer for each event**.  
It leverages vectorized NumPy operations (`bincount`, mapping, reshaping) to efficiently compute average energies without explicit Python loops.

---

### Parameters
- **filename** (*str*, optional):  
  Path to the input HDF5 file containing calorimeter event data.  
  Default: `"hgcal_electron_data_0001.h5"`  

---

### Returns
- **pd.DataFrame**:  
  A DataFrame where:
  - Each row corresponds to a single event.
  - Columns include:
    - `"event_no"`: The event index (0-based).
    - `"z_i_average_energy"`: The average energy deposited in the *i-th layer* (based on sorted unique z-values).

---

### Processing Steps
1. **Load Data**: Extracts `nhits`, `rechit_z`, and `rechit_energy` arrays from the input file.  
2. **Prepare Metadata**:
   - Ensures `nhits` is cast to integer (avoids type errors).
   - Identifies unique sorted `z` positions → defines detector layers.  
3. **Index Mapping**:
   - Creates `event_indices` (mapping each hit to its event).
   - Maps `z` values to corresponding layer indices.  
   - Constructs **linear indices** for efficient aggregation.  
4. **Aggregation**:
   - Computes **sum of energies per (event, layer)**.  
   - Computes **hit counts per (event, layer)**.  
   - Divides sums by counts → **average energies**.  
5. **Reshape and Format**:
   - Reshapes results into `(n_events, n_layers)`.  
   - Creates a DataFrame with descriptive column names.  
   - Inserts `"event_no"` as the first column.  

---

### Example Output (Schema)
```text
   event_no  z_1_average_energy  z_2_average_energy  ...  z_26_average_energy  z_27_average_energy  z_28_average_energy
0         0           29.806881           14.598131  ...             2.974839             3.965091             4.667946
1         1           15.072347           22.016330  ...             3.932254             2.590214             2.942818
2         2           20.345368           17.638254  ...             4.094181            11.780640             5.602276
3         3           10.748526           34.254235  ...             3.351710             1.097838             0.000000
4         4            4.632542           11.599174  ...             2.674365             4.874276             3.127094
```

- And has a dimension [len(event) rows x 29 columns]


```python
def prepare_event_layer_dataframe_cpu(filename="hgcal_electron_data_0001.h5"):
    """
    Prepare a DataFrame with average energy per layer for each event.

    Args:
        filename (str, optional): Name of the HDF5 file. Defaults to "hgcal_electron_data_0001.h5".

    Returns:
        pd.DataFrame: DataFrame with average energy per layer for each event.
    """
```

## Line-by-Line Explanation of `prepare_event_layer_dataframe_cpu`

This section breaks down each line of the function and explains its role with examples.

---

### Load the dataset
```python
dataset = load_data(filename)
```
- Calls the custom function `load_data` to read the HDF5 file.

### Extract relevant arrays

```python
nhits = dataset["nhits"]
zs, energies = dataset["rechit_z"], dataset["rechit_energy"]
```

- Assigns number of hits per event (`nhits`), hit positions(`zs`), and energy values(`energies`).

### Identify unique detector layers

```python
unique_zs = np.sort(np.unique(zs))
n_layers = len(unique_zs)
```

- Find unique sorted `z` positions → defines detector layers.

### Event index for each hit

```python
event_indices = np.repeat(np.arange(n_events), nhits)
```

- Expands the event numbers so each hit is linked to its event.

### Map z-values to layer indices

```python
z_to_col = {z: i for i, z in enumerate(unique_zs)}
col_indices = np.array([z_to_col[z] for z in zs])
```

- Creates a dictionary mapping each unique `z` to a layer index.

### Construct linear indices

```python
linear_idx = event_indices * n_layers + col_indices
```

- Flattens `(event,layers)` pairs into single indices for np.bitcount.

### Aggregate energies per (event, layer)

```python
energy_sum = np.bincount(linear_idx, weights=energies, minlength=n_events * n_layers)
hit_count = np.bincount(linear_idx, minlength=n_events * n_layers)
```

- `energy_sum`: sums energy for each `(event, layer)` bin.
- `hit_count`: counts hits for each `(event, layer)` bin.

### Compute average energies

```python
avg_energy = energy_sum / np.maximum(hit_count, 1)
```
- Divides energy sums by hit_counts to get averages.
- uses `np.maximum(hit_count,1) to avoide division by 0.

### Reshape into event × layer matrix

```python
avg_energy_matrix = avg_energy.reshape(n_events, n_layers)
```

- Reshapes flat array into a 2D matrix.

### Create Column names 

```python
column_names = [f"z_{i+1}_average_energy" for i in range(n_layers)]
```

- Generates discriptive column headers.

### Build DataFrame

```python
column_names = [f"z_{i+1}_average_energy" for i in range(n_layers)]
```

- Builds final DataFrame:
    - Adds `event_no` as column.

## Prepare Event Layer Dataframe (GPU version)

<span style="color:red">Not ready yet</span>

## Plot Average Energy Per Layer

- `plot_average_energy_per_layer` takes a random event from the dataframe containing the average energy per layer for each event and plots it.

```python
def plot_average_energy_per_layer(df):
    """Plot average energy per layer for a random sample of events.
    Args:
        df (pd.DataFrame): DataFrame containing average energy per layer for each event.
    """
```