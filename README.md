# PDPO

This folder contains the implementation of our algorithm **PDPO**.  
1. Install the necessary packages by creating a conda environment using the yml file
```
# Create conda environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate pdpo
```
2. Replace the contents of the file **ode.py** in the _torchdyn_ library with the contents of the file **/parametric_pushforward/ode_file_substitute.py.** *** PDPO WILL NOT WORK IF THIS STEP IS OMITTED.***
