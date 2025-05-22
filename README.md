# PDPO
## Setup PDPO
This folder contains the implementation of our algorithm **PDPO**.  
1. Install the necessary packages by creating a conda environment using the yml file
```
# Create conda environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate pdpo
```
2. Replace the contents of the file **ode.py** in the _torchdyn_ library with the contents of the file **/parametric_pushforward/ode_file_substitute.py.** 	***PDPO WILL NOT WORK IF THIS STEP IS OMITTED.***

## Visualize trained models.

-To visualize the solutions in the section _Obstacle avoidance with mean-field interaction_ , access the notebooks folder and open _visualize_density_path.ipynb_. Find the different experiment names for the variable `exp_dir` in the _experiments_ folder. 
-To compare our solution with other methods, open the file _compare_results.ipynb_. The rest of the methods can also be loaded in this jupiter notebook. 

## Train a model


