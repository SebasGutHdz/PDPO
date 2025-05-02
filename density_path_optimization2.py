"""
This module implements density path optimization for optimal transport problems.
It provides functionality to optimize paths between source and target distributions
using parametric pushforward maps and various potential functions.

The main components include:
- Experiment configuration and setup
- Model initialization and training
- Visualization of optimization results
- Integration with Weights & Biases for experiment tracking
"""

from pathlib import Path
project_root = Path(__file__).parent.absolute()
import sys

# sys.path.append(str(project_root))

import os

import yaml

import torch
import matplotlib.pyplot as plt


import wandb

from ema_pytorch import EMA
from datetime import datetime
import seaborn as sns

# from parametric_pushforward.obstacles import obstacle_cost_stunnel, obstacle_cost_vneck, obstacle_cost_gmm,congestion_cost,geodesic,quadartic_well
from parametric_pushforward.opinion import PolarizeDyn
from parametric_pushforward.spline2 import Assemble_spline
from parametric_pushforward.visualization import path_visualization_snapshots,disimilarity_snapshots,plot_hist
from parametric_pushforward.setup_density_path_problem2 import initialize_experiment,load_boundary_models,setup_prior,get_activation,get_potential_functions,opinion_dynamics_setup,setup_optimizers
from parametric_pushforward.parametric_mlp import MLP,order_state_to_tensor
import parametric_pushforward.data_sets as data_sets

from geomloss import SamplesLoss


def run_experiments(config_path):
    """Run the density path optimization experiment.
    
    This function manages the entire optimization process including:
    1. Loading and setting up the experiment configuration
    2. Initializing models and optimizers
    3. Running the optimization loop with path and coupling optimization
    4. Generating visualizations and logging results
    
    Args:
        config_path (str): Path to the YAML configuration file containing experiment parameters
        
    The configuration file should specify:
    - Data source and target distributions
    - Model architecture parameters
    - Optimization parameters
    - Potential functions
    - Visualization settings
    - WandB logging configuration
    """

    with open(config_path,'r') as f:
        config = yaml.safe_load(f)

    # Initialize WandB with custom run name including potentials
    # Generate run name with fallback defaults
    source_name = config['data']['source']['name']
    target_name = config['data']['target']['name']
    potentials = "_".join(config['potential_functions'])
    
    # Default run name if not specified in config
    default_run_name = f"{source_name}_to_{target_name}_pot_{potentials}"
    
    # Get run name from config if it exists, otherwise use default
    run_name = config.get('wandb', {}).get('run_name', default_run_name)
    
    # Replace template variables if they exist in the run name
    if isinstance(run_name, str):
        run_name = run_name.replace("${data.source.name}", source_name)
        run_name = run_name.replace("${data.target.name}", target_name)
        run_name = run_name.replace("${potential_functions}", potentials)

    # Create experiment output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("experiments", f"{run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save a copy of the config file
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize wandb with safe defaults
    wandb_config = config.get('wandb', {})
    wandb.init(
        project=wandb_config.get('project', 'density_path_optimization'),
        entity=wandb_config.get('entity', None),
        name=run_name,
        group=wandb_config.get('group', 'density_path_experiments'),
        config=config
    )
    
    # Setup device
    device = torch.device(config['device'])
    
    # Initialize experiment
    initialize_experiment(config['experiment_seed'])
    
    # Setup prior distribution
    prior = setup_prior(config, device)
    arch = [
        config['architecture']['input_dim'],
        config['architecture']['hidden_dim'],
        config['architecture']['num_layers'],
        torch.nn.Softplus()
    ]
    # Initialize dummy model
    model0 = MLP(arch, time_varying=config['architecture']['time_varying']).to(device)
    # Zero initialization for theta0
    theta0 = order_state_to_tensor(model0.state_dict())
    # Pretrained theta1
    state1 = load_boundary_models(config, device)
    theta1 = order_state_to_tensor(state1)
    
        
        
    
    # Setup architecture
    activation = get_activation(config['architecture']['activation'])
    arch = [
        config['architecture']['input_dim'],
        config['architecture']['hidden_dim'],
        config['architecture']['num_layers'],
        activation
    ]

    # Setup opinion dynamics
    if config.get('opinion_dynamics', {}).get('active', False):
        opinion_dynamics = opinion_dynamics_setup(config)
        ke_modifier = [PolarizeDyn(opinion_dynamics).to(device)]
    else:
        ke_modifier = None

    
    
    # Initialize spline
    spline, _ = Assemble_spline(
        theta0=theta0,
        theta1=theta1,
        arch=arch,
        data0=config['data']['source']['name'],
        data1=config['data']['target']['name'],
        ke_modifier=ke_modifier,
        potential= get_potential_functions(config['potential_functions']),#config['potential_functions'],
        number_of_knots=config['spline']['num_collocation'],
        spline=config['spline']['type'],
        device=device,
        prior_dist=prior
    )
    # Setup sigma for Fisher Information
    spline.sigma = config['coefficients_potentials']['sigma']
    # Setup optimizers and schedulers
    (optimizer_path, scheduler_path), (optimizer_coupling, scheduler_coupling) = setup_optimizers(spline, config)
    
    # Setup EMA
    ema = EMA(
        spline,
        beta=config['ema']['beta'],
        update_after_step=config['ema']['update_after_step'],
        update_every=config['ema']['update_every']
    )
    
    # Generate fixed samples for visualization
    z_ = prior.sample((config['optimization']['num_samples'],)).to(device)
    
    # Training loop
    lagrangian_history = []
    ke_history = []
    potential_history = []
    bd0_history = []
    bd1_history = []
    bd0_distance = []
    bd1_distance = []
    checkpoint_path = os.path.join(checkpoints_dir, f"initial.pth")
    torch.save({
                'direct_model': spline.state_dict(),
                'ema_model':ema.model.state_dict()},
                checkpoint_path
            )

    # For non zero potentials we recommend a geodesic warmup
    if config['optimization']['geodesic_warmup']:
        geodesic_optimizer = torch.optim.Adam(
            [spline.knots],
            lr=1e-3
        )
        spline.geodesic_warmup(geodesic_optimizer,num_epochs=config['optimization']['geodesic_warmup_steps'])
        checkpoint_path = os.path.join(checkpoints_dir, f"geo_inital.pth")
        torch.save({
                    'direct_model': spline.state_dict(),
                    'ema_model':ema.model.state_dict()},
                    checkpoint_path
                )

    loss = SamplesLoss(loss = 'sinkhorn',p =2 ,blur = 0.05)
    comp_bd = 5000

    for experiment in range(len(config['optimization']['optimization_steps'])):
        print(f'Experiment {experiment + 1}')


        
        for i in range(config['optimization']['optimization_steps'][experiment]):

            # Visualization and logging
        
            if i == 0 or i == config['optimization']['optimization_steps'][experiment] - 1 or (i + 1) % (config['optimization']['optimization_steps'][experiment] // 5) == 0:

                s = torch.linspace(0, 1, 30).to(device)
                interpolation = ema(s)
                # Create and save visualization
                plt.figure(figsize=(10, 10))
                
                samples_path = path_visualization_snapshots(
                    interpolation=interpolation,
                    arch=arch,
                    spline=spline,
                    x0 = config['visualization']['plot_bounds']['x_min'],
                    y0 = config['visualization']['plot_bounds']['y_min'],
                    x1 = config['visualization']['plot_bounds']['x_max'],
                    y1 = config['visualization']['plot_bounds']['y_max'],
                    num_samples=config['visualization']['num_plot_samples'],
                    time_steps=config['visualization']['num_time_steps'],
                    solver=config['visualization']['solver'],
                    z=z_,
                    num_contour_points=100
                )
                
                # Log to WandB
                wandb.log({
                    'path_plot': wandb.Image(plt),
                })
                plt.savefig(os.path.join(figures_dir, f"path_plot_{i}.png"))
                plt.close()

                # For opinion dyanmics visualize the disimilarity
                if config.get('opinion_dynamics',{}).get('active', False):
                    plt.figure(figsize=(10, 10))
                    disimilarity_snapshots(samples_path)
                    # Log to WandB
                    wandb.log({
                        'dissimilarity_plot': wandb.Image(plt),
                    })
                plt.close()

                checkpoint_path = os.path.join(checkpoints_dir, f"spline.pth")
                torch.save({
                    'direct_model': spline.state_dict(),
                    'ema_model':ema.model.state_dict()},
                    checkpoint_path
                )

                z_comp_bd = prior.sample((comp_bd,)).to(device= device)
                samples_bd0 = torch.from_numpy(data_sets.inf_train_gen(config['data']['source']['name'],batch_size= comp_bd,dim= config['architecture']['input_dim'])).to(device)
                samples_bd1 = torch.from_numpy(data_sets.inf_train_gen(config['data']['target']['name'],batch_size= comp_bd,dim = config['architecture']['input_dim'])).to(device)

                bd0_generated = spline.push_forward(spline.x0.flatten(),z = z_comp_bd)
                bd1_generated = spline.push_forward(spline.x1.flatten(),z=z_comp_bd)
                
                bd0_distance.append(loss(samples_bd0,bd0_generated).detach().cpu().item())
                bd1_distance.append(loss(samples_bd1,bd1_generated).detach().cpu().item())





            # Path optimization
            print('Optimizing path...')
            # Generate fixed samples to optimize the path, if config['optimization']['batch_size'] = None
            # random samples are going to be generated every optimization step. This can lead to  unstable 
            # optimization of the path.
            x0_ = prior.sample((config['optimization']['batch_size'],)).to(device)
            #epochs, optimizer, scheduler, t_partition, ema=None, t_node=10, bs=1000, x0=None
            outputs = spline.optimize_path(
                epochs=config['optimization']['path']['steps'][experiment],
                optimizer=optimizer_path,
                scheduler=scheduler_path,
                t_partition=config['spline']['t_partition'],
                ema=ema,
                t_node=config['optimization']['t_node'],
                bs=config['optimization']['batch_size'],
                x0=x0_
            )
            
            lagrangian_history.append(outputs['lagrangian'])
            ke_history.append(outputs['kinetic'])
            potential_history.append(outputs['potential'])
            for lag,(ke,pot) in zip(outputs['lagrangian'],zip(outputs['kinetic'],outputs['potential'])):
                wandb.log({
                    'lagrangian': lag,
                    'kinetic energy': ke,
                    'potential energy': pot
                })
            
            # Coupling optimization
            print('Optimizing coupling...')
            # Keep the prior samples whose path has been optimized
            # epochs, optimizer, scheduler, t_partition, ema=None, t_node=10, 
            # bs=1000, weight_bd=1000, x0=None        
            outputs = spline.optimize_coupling(
                epochs=config['optimization']['coupling']['steps'][experiment],
                optimizer=optimizer_coupling,
                scheduler=scheduler_coupling,
                t_partition=config['spline']['t_partition'],
                ema=ema,
                t_node=config['optimization']['t_node'],
                bs=config['optimization']['batch_size'],
                weight_bd=config['optimization']['weight_boundary'],
                x0=x0_               
            )

            bd0_history.append(outputs['bd_0'])
            bd1_history.append(outputs['bd_1'])
            for bd0,bd1 in zip(outputs['bd_0'],outputs['bd_1']):
                wandb.log({
                    'Loss_fn bd0': bd0 ,
                    'Loss_fn bd1': bd1
                })
                       
            
        plot_hist(lagrangian_history, potential_history,bd0_distance,bd1_distance, figures_dir)
        wandb.finish()

if __name__ == "__main__":
    """
    Entry point for running the density path optimization experiment.
    Specify the configuration file to use for the experiment.
    """
    # name_experiment  = 'configs_2D_gauss0_d_gauss1_d_SB.yaml'
    # name_experiment  = 'configs_2D_gauss0_d_gauss1_d_geo.yaml'
    # name_experiment  = 'configs_10D_gauss0_d_gauss1_d_geo.yaml'
    # name_experiment  = 'configs_10D_gauss0_d_gauss1_d_SB.yaml'
    # name_experiment  = 'configs_50D_gauss0_d_gauss1_d_geo.yaml'
    # name_experiment  = 'configs_50D_gauss0_d_gauss1_d_SB.yaml'
    # name_experiment  = 'configs_100D_gauss0_d_gauss1_d_geo.yaml'
    # name_experiment  = 'configs_50D_gauss0_d_gauss1_d_SB.yaml'
    # name_experiment  = 'configs_8gmm_half_std.yaml'
    name_experiment  = 'configs_8gmm_4gmm2.yaml'
    # name_experiment  = 'configs_2D_vneck.yaml'
    # name_experiment  = 'configs_2D_scurve.yaml'
    # name_experiment  = 'configs_opinion_2D.yaml'
    # name_experiment  = 'configs_opinion_1000D.yaml'
    dir_ = project_root / 'configs' / 'density_path_problems' / name_experiment
    run_experiments(str(dir_)) 








