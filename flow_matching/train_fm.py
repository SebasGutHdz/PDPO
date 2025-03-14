import sys
import os

projec_root = os.path.abspath(os.path.join(os.getcwd(),'../'))
sys.path.append(projec_root)

import yaml
import torch
from pathlib import Path
import parametric_pushforward.data_sets as data_sets
from parametric_pushforward.parametric_mlp import MLP,ParameterizedMLP,ParameterizedWrapper,order_state_to_tensor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm



ACTIVATION_FNS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'swish': nn.SiLU(),
    'sin': torch.sin()
}


class Config:

    def __init__(self,config_path):

        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)

    def get_model(self):
        '''
        Initialize model based on config file
        '''
        model_config = self.config['model']
        activation_fn = ACTIVATION_FNS[model_config['activation_fn']]
        arch = [model_config['input_dim'],model_config['hidden_dim'],model_config['num_layers'],activation_fn]
        model = MLP(arch,time_varying=model_config['time_varying']).to(self.config['training']['device'])

        return model
    
    def get_data_set(self):
        '''
        Initialize data set based on config file
        '''
        data_config = self.config['data']
        data_set = data_sets.inf_train_gen(data_config['type'],batch_size = data_config['total_data'],dim= self.config['model']['input_dim'])

        data_set = TensorDataset(torch.tensor(data_set).float().to(self.config['training']['device']))
        dataloader = DataLoader(data_set,batch_size = self.config['training']['batch_size'],shuffle = True)


        return dataloader

    def get_optimizer(self,model):
        '''
        Initialize optimizer based on config file
        '''
        lr = self.config['training']['learning_rate']
        optimizer = torch.optim.Adam(model.parameters(),lr = lr)

        return optimizer
    
    def save_checkpoint(self,model,optimizer,epoch,loss,path):
        '''
        Save model checkpoint
        '''
        checkpoint = {
            'epoch': epoch,
            # 'arch': [model.in_features,model.hidden_dim,model.num_layers,],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        torch.save(checkpoint,path)
        





def loss_flow_matching(model,data_set,device,batch_size = 1000,dim = 2):
    '''
    Input:
        model: torch.nn.Module
        data_set: torch.utils.data.DataLoader
        device: str
        batch_size: int
        dim: int
    Output:
        loss: torch.tensor
    '''
    
    z = torch.randn(batch_size,dim).to(device)
    # Sample location and conditional flow
    t = torch.rand(batch_size).to(device)
    zt = data_set*t[:,None] + (1-t[:,None])*z
    xt = 0.1*torch.randn(batch_size,dim).to(device) + zt
    ut = data_set-z
    # Compute the model prediction
    vt = model(torch.cat([xt,t[:,None]],dim = -1))
    loss = torch.mean(torch.sum((vt-ut)**2,dim = -1))

    return loss

def train_flow_matching(config_path):

    cfg = Config(config_path)

    model = cfg.get_model()
    dataloader = cfg.get_data_set()
    optimizer = cfg.get_optimizer(model)
    device = cfg.config['training']['device']

    # Checkpoint path
    path = cfg.config['training']['checkpoint_dir']+cfg.config['data']['type']+f'[{cfg.config["model"]["input_dim"]},{cfg.config["model"]["hidden_dim"]},{cfg.config["model"]["num_layers"]},{cfg.config["model"]["activation_fn"]}]'
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(exist_ok = True)

    #Training loop
    prange = tqdm(range(cfg.config['training']['n_epochs']))

    for epoch in prange:

        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()
            loss = loss_flow_matching(model,batch[0],device,batch_size = cfg.config['training']['batch_size'],dim = cfg.config['model']['input_dim'])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            num_batches += 1
            total_loss += loss.item()

        avg_loss = total_loss/num_batches
        prange.set_description(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')

        # Save checkpoint
        if (epoch+1) % cfg.config['training']['save_interval'] == 0:
            cfg.save_checkpoint(model,optimizer,epoch,avg_loss,checkpoint_dir/f'checkpoint_{epoch}.pth')
        if epoch == cfg.config['training']['n_epochs']-1:
            cfg.save_checkpoint(model,optimizer,epoch,avg_loss,checkpoint_dir/f'final.pth')
    # Compare learned model with true samples

    return model

if __name__ == '__main__':
    train_flow_matching('configs/configs_train_fm.yaml')
    

    
