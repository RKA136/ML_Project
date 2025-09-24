# ML Project

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