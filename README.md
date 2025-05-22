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

* To visualize the solutions in the section _Obstacle avoidance with mean-field interaction_ , access the notebooks folder and open _visualize_density_path.ipynb_. Find the different experiment names for the variable `exp_dir` in the _experiments_ folder. 

* To compare our solution with other methods' solutions, open the file _compare_results.ipynb_. The rest of the methods can also be loaded in this jupiter notebook. 

## Train a model
For all experiments, except GMM, open the file _density_path_optimization.py_. At the bottom of the file, you can uncomment the experiment you want to run. To modify the experimental setup of an example, locate the yaml files in the _configs/density_path_problems_ folder.

For the GMM experiment, use file  _density_path_optimization2.py_.

## Visualize a new model

Follow the same steps as before, in the _experiments_ folder you will find the new experiment folder with the following naming convention `source_density_name-_to_target-density-name_name-of-obstable_time-stamp.`



