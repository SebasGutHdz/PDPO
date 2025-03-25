'''
Flow Matching Training Implementation

This module implements training for flow matching, a method for learning NODEs.
Flow matching directly learns the velocity field of a continuous-time transformation between
distributions by minimizing the L2 distance between the predicted and target velocities.

Key Components:
1. Configuration Management:
   - Handles model architecture, training parameters, and data settings
   - Manages checkpointing and model saving

2. Training Loop:
   - Implements flow matching loss computation
   - Handles batch processing and optimization
   - Tracks and saves training progress

3. Model Architecture:
   - Supports various activation functions
   - Configurable MLP-based architecture
   - Time-varying vector field implementation

Reference: https://arxiv.org/abs/2210.02747
'''

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))


import yaml
import torch
from pathlib import Path
import parametric_pushforward.data_sets as data_sets
from parametric_pushforward.parametric_mlp import MLP,Sin

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Dictionary mapping activation function names to their PyTorch implementations
ACTIVATION_FNS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'swish': nn.SiLU(),
    'sin': Sin()
}

class Config:
    '''
    Configuration manager that handles model setup, data loading, and training parameters.
    Loads settings from a YAML file and provides methods for initialization.
    '''
    def __init__(self, config_path: str | Path) -> None:
        try:
            with open(config_path,'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
    def get_model(self)-> nn.Module:
        '''
        Initialize model based on config file.
        Creates an MLP with specified architecture and activation function.
        '''
        model_config = self.config['model']
        activation_fn = ACTIVATION_FNS[model_config['activation_fn']]
        arch = [model_config['input_dim'],
                model_config['hidden_dim'],
                model_config['num_layers'],
                activation_fn]
        model = MLP(arch, time_varying=model_config['time_varying']).to(self.config['training']['device'])
        
        
        return model
    
    def get_data_set(self):
        '''
        Initialize dataset based on config file.
        Creates a DataLoader with specified batch size and data type.
        '''
        data_config = self.config['data']
        # Generate infinite training data
        data_set = data_sets.inf_train_gen(data_config['type'],
                                         batch_size=data_config['total_data'],
                                         dim=self.config['model']['input_dim'])
        
        # Convert to PyTorch dataset and create dataloader
        data_set = TensorDataset(torch.tensor(data_set).float().to(self.config['training']['device']))
        dataloader = DataLoader(data_set,batch_size=self.config['training']['batch_size'],shuffle=True)
        return dataloader

    def get_optimizer(self,model: nn.Module)->torch.optim.Optimizer:
        '''
        Initialize optimizer based on config file.
        Currently uses Adam optimizer with specified learning rate.
        '''
        lr = self.config['training']['learning_rate']
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer)
        return optimizer
    
    def save_checkpoint(self,model,optimizer,epoch,loss,path):
        '''
        Save model checkpoint including model state, optimizer state, and training progress.
        
        Args:
            model: Neural network model
            optimizer: Optimizer instance
            epoch: Current training epoch
            loss: Current loss value
            path: Path to save checkpoint
        '''
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint,path)

def loss_flow_matching(
        model: nn.Module,
        data_set: torch.Tensor,
        device: torch.device,
        batch_size=1000,
        dim=2):
    '''
    Compute flow matching loss between predicted and target velocities.
    
    The loss measures the L2 distance between:
    1. The predicted velocity field from the model
    2. The target velocity field (straight-line path between source and target)
    
    Args:
        model: Neural network model
        data_set: Target data samples
        device: Computing device (CPU/GPU)
        batch_size: Number of samples per batch
        dim: Dimension of the data
    Returns:
        loss: Average L2 distance between predicted and target velocities
    '''
    batch_size = data_set.shape[0]
    # Sample from prior distribution
    z = torch.randn(batch_size,dim).to(device)
    
    # Sample interpolation time and compute intermediate points
    t = torch.rand(batch_size).to(device)
    
    zt = data_set*t[:,None] + (1-t[:,None])*z  # Linear interpolation
    xt = torch.randn(batch_size,dim).to(device)*0.1 + zt  # Add noise
    
    # Compute target velocity (straight line to target)
    ut = data_set-z
    
    # Compute model's predicted velocity
    vt = model(torch.cat([xt,t[:,None]],dim=-1))
    
    # Compute MSE loss between predicted and target velocities
    loss = torch.mean(torch.sum((vt-ut)**2,dim=-1))
    return loss

def train_flow_matching(config_path):
    '''
    Main training loop for flow matching.
    
    Args:
        config_path: Path to configuration YAML file
    Returns:
        model: Trained neural network model
    '''
    # Initialize configuration and components
    cfg = Config(config_path)
    model = cfg.get_model()
    dataloader = cfg.get_data_set()
    optimizer = cfg.get_optimizer(model)
    device = cfg.config['training']['device']

    # Setup checkpoint directory
    path = cfg.config['training']['checkpoint_dir']+cfg.config['data']['type']+\
           f'[{cfg.config["model"]["input_dim"]},{cfg.config["model"]["hidden_dim"]},'+\
           f'{cfg.config["model"]["num_layers"]},{cfg.config["model"]["activation_fn"]}]'
    path = str(project_root)+'/'+path
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop with progress bar
    prange = tqdm(range(cfg.config['training']['n_epochs']))
    for epoch in prange:
        total_loss = 0
        num_batches = 0

        # Batch training
        for batch in dataloader:
            optimizer.zero_grad()
            loss = loss_flow_matching(model, batch[0], device,
                                    batch_size=cfg.config['training']['batch_size'],
                                    dim=cfg.config['model']['input_dim'])
            loss.backward()
            
            
            
            optimizer.step()
            num_batches += 1
            total_loss += loss.item()

        # Update progress bar with current loss
        avg_loss = total_loss/num_batches
        prange.set_description(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')

        # Save periodic checkpoints
        if (epoch+1) % cfg.config['training']['save_interval'] == 0:
            cfg.save_checkpoint(model,optimizer,epoch,avg_loss,
                              checkpoint_dir/f'checkpoint_{epoch}.pth')
        
        # Save final model
        if epoch == cfg.config['training']['n_epochs']-1:
            cfg.save_checkpoint(model,optimizer,epoch,avg_loss,
                              checkpoint_dir/f'final.pth')
            config_save_path = checkpoint_dir/f'config.yaml'
            with open(config_save_path,'w') as f:
                yaml.dump(cfg.config,f,default_flow_style=False)

    return model

if __name__ == '__main__':    
    train_flow_matching(str(project_root)+'/configs/fm_training/configs_train_fm.yaml')
    
    

    
