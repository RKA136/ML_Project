# ML Project

# Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Functions](#functions)

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
    dataset = {}
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config["data_dir"]
    filename = os.path.join(data_dir, filename)
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            dataset[key] = f[key][:]
    return dataset
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
    dataset = load_data(filename)
    nhits = dataset["nhits"]
    xs, ys, zs, energies = dataset["rechit_x"], dataset["rechit_y"], dataset["rechit_z"], dataset["rechit_energy"]
    targets = dataset["target"]
    
    i = event_index
    start, end = int(np.sum(nhits[:i])), int(np.sum(nhits[:i+1]))
    x, y, z, e = xs[start:end], ys[start:end], zs[start:end], energies[start:end]
    true_E = targets[i]
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    sizes = 5 # scale marker sizes with energy
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=e,
            colorscale='Viridis',
            colorbar=dict(title="Energy (MIP)"),
            opacity=0.7
        ),
        text=[f"E={ee:.2f} MIP" for ee in e],  # hover text
    )])
    fig.update_layout(
        title=f"Interactive 3D Shower (Event {i}, True E = {true_E:.1f} GeV)",
        scene=dict(
            xaxis_title='x [cm]',
            yaxis_title='y [cm]',
            zaxis_title='z [cm]'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig_path = os.path.join(figures_dir, f"event_display_{i}.html")
    fig.show()
    fig.write_html(fig_path)
    print(f"Saved interactive figure as {fig_path}")
```

## Hits per Event

- `hits_per_event` plots the number of hits in each event as a histogram and saves the plot in the figures folder.

```python
def hits_per_event(filename="hgcal_electron_data_0001.h5"):
    """Plot a histogram of the number of hits per event in the dataset.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    """
    dataset = load_data(filename)
    nhits = dataset["nhits"]
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nhits, bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Number of Hits per Event")   
    ax.set_xlabel("Number of Hits")
    ax.set_ylabel("Number of Events")
    hist_path = os.path.join(figures_dir, "hits_per_event.png")
    plt.savefig(hist_path)
    plt.close()
```

## True Energy Distrinution

- `true_energy_distribution` plots the true energies in a histogram and saves the figure in the figures folder.

```python
def true_energy_distribution(filename="hgcal_electron_data_0001.h5"):
    """Plot a histogram of the true energy distribution of events in the dataset.
    Args:
        filename (str): Name of the HDF5 file containing the dataset.
    """
    dataset = load_data(filename)
    targets = dataset["target"]
    with open("config.json", "r") as f:
        config = json.load(f)
    figures_dir = config["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(targets, bins=30, color='lightgreen', edgecolor='black')
    ax.set_title("Histogram of True Energy Distribution")   
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Number of Events")
    hist_path = os.path.join(figures_dir, "true_energy_distribution.png")
    plt.savefig(hist_path)
    plt.close()
```