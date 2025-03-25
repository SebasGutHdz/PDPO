from pathlib import Path
path_root = Path(__file__).parent.parent.absolute()
print(str(path_root))

import torch
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from obstacles import obstacle_cost_stunnel, obstacle_cost_vneck, obstacle_cost_gmm,congestion_cost,geodesic,quadartic_well

import numpy as np
import os


def initialize_experiment(seed = 0):
    '''Initialize random state'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_boundary_models(config,device):
    def make_arch_str(dims,act_fn):
        # Remove quotes from act_fn
        act_fn = act_fn.replace("'","")
        return f"[{dims[0]},{dims[1]},{dims[2]},{act_fn}]"
    # Load boundary models from checkpoints
    model_path = lambda name,arch,ckpt: os.path.join(
        config['data']['model_dir'],
        f"{name}{arch}/{ckpt}.pth"
    )
    arch_dims = [
        config['architecture']['input_dim'],
        config['architecture']['hidden_dim'],
        config['architecture']['num_layers']
    ]
    arch_str = make_arch_str(arch_dims,config['architecture']['activation'])
    #Load bd0 model
    bd0_path = model_path(
        config['data']['source']['name'],
        arch_str,
        config['data']['source']['checkpoint'],
    )
    bd0_path_str  = str(path_root)+str(bd0_path)
    bd0_path = Path(bd0_path_str)
    

    bd0_parameter = torch.load(bd0_path,map_location=device)['model_state_dict']

    #Load bd1 model
    bd1_path = model_path(
        config['data']['target']['name'],
        arch_str,
        config['data']['target']['checkpoint']
    )

    bd1_path_str  = str(path_root)+str(bd1_path)
    bd1_path = Path(bd1_path_str)

    bd1_parameter = torch.load(bd1_path,map_location=device)['model_state_dict']

    return bd0_parameter,bd1_parameter

def setup_prior(config,device):
    '''Setup prior distribution'''
    assert config['prior']['type'] == 'gaussian',f"Prior type ({config['prior']['type']}) must be 'gaussian'."
    assert config['prior']['dimension'] == config['architecture']['input_dim'], f"Prior dimension ({config['prior']['dimension']}) must match network input dimension ({config['architecture']['input_dim']})"
    prior = MultivariateNormal(torch.zeros(config['architecture']['input_dim']).to(device),torch.eye(config['architecture']['input_dim']).to(device))
    
    return prior
def get_activation(name):
    '''Get Activation function by name'''
    activations = {
        'softplus': F.softplus,
        'relu': F.relu,
        'tanh': F.tanh,
        'sin': lambda x: torch.sin(x)
    }
    return activations.get(name,F.softplus)

def get_potential_functions(names):
    """Get potential functions by name."""
    potential_functions = {
        'obstacle_cost_stunnel': obstacle_cost_stunnel,
        'obstacle_cost_vneck': obstacle_cost_vneck,
        'obstacle_cost_gmm': obstacle_cost_gmm,
        'congestion_cost': congestion_cost,
        'geodesic': geodesic,
        'quadartic_well': quadartic_well,
        # Add more potential functions as needed
    }
    potentials = []
    if 'entropy' in names:
        potentials += ['entropy']
        names.remove('entropy')
    if 'fisher_information' in names:
        potentials += ['fisher_information']
        names.remove('fisher_information')
    potentials += [potential_functions[name] for name in names]
    
    return potentials

def opinion_dynamics_setup(config):
    """Setup opinion dynamics parameters."""
    class Configs_Opinion():

        def __init__(self):

            self.D = config['architecture']['input_dim']
            self.S = config['opinion_dynamics']['S']
            self.strength = config['opinion_dynamics']['strength']
            self.m_coeff = config['opinion_dynamics']['m_coeff']

    configs_opinion = Configs_Opinion()
        
    return configs_opinion

def setup_optimizers(spline,config):
    """Set up optimizers"""
    # Dictionary configurations optimizer
    opt_config = config['optimization']
    # Set up path optimizer
    # spline.knots are not a list! torch.optim(params = ) requires a list
    optimizer_path = torch.optim.Adam(params= [spline.knots],
                                    lr = opt_config['path']['learning_rate'],
                                    betas = opt_config['path'].get('betas',(0.9,0.999)),
                                    eps = opt_config['path'].get('eps',1e-8)
    )
    scheduler_path = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer= optimizer_path,
        T_0 = opt_config['path']['scheduler'].get('T_0',200),     # Initial restartin period
        T_mult=opt_config['path']['scheduler'].get('T_mult', 2),  # Increase restart period
        eta_min=opt_config['path']['scheduler'].get('eta_min', 1e-6)  # Minimum learning rate
        )
    # Coupling optimizer
    optimizer_coupling = torch.optim.AdamW(
        [spline.x0, spline.x1],
        lr=opt_config['coupling']['learning_rate'],
        weight_decay=opt_config['coupling']['weight_decay']
    )
    
    scheduler_coupling = torch.optim.lr_scheduler.StepLR(
        optimizer_coupling,
        step_size=opt_config['coupling']['scheduler']['step_size'],
        gamma=opt_config['coupling']['scheduler']['gamma']
    )

    return (optimizer_path,scheduler_path),(optimizer_coupling,scheduler_coupling)

    
    
            
    