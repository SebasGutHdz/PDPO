'''
\https://github.com/SebasGutHdz/PDPO/blob/main/parametric_pushforward/parametric_mlp.py
This module implements various Multi-Layer Perceptron (MLP) architectures and wrappers
for the parametric pushforward optimization framework.

Key Components:
1. Activation Functions:
   - Sin: Sinusoidal activation
   - Swish: Self-gated activation (x * sigmoid(x))

2. Neural Network Architectures:
   - MLP: Standard MLP with configurable architecture
   - ParameterizedMLP: MLP where weights/biases are provided as a flat parameter vector
   
3. Wrapper Classes:
   - torch_wrapper: Adapts models to torchdyn format
   - ParameterizedWrapper: Wraps parameterized models for ODE solving

The networks support time-varying behavior by optionally including time as an input dimension.
'''


import torch as torch
import torch.nn.functional as F
import torch.nn as nn




# MLP model
class Sin(nn.Module):
    '''
    Sinusoidal activation function layer.
    Useful for modeling periodic functions or continuous flows.
    '''
    def forward(self, x):
        return torch.sin(x)

class Swish(nn.Module):
    '''
    Swish activation function: x * sigmoid(x)
    Combines the properties of ReLU and sigmoid for smoother gradients.
    '''
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(torch.nn.Module):
    def __init__(self, arch, time_varying=True):
        '''
        Standard MLP implementation with configurable architecture.
        
        Input:
            arch: [dim, width, num_layers, activation_fn]
                - dim: Input/output dimension
                - width: Hidden layer width
                - num_layers: Total number of layers
                - activation_fn: Optional activation function (defaults to ReLU)
            time_varying: If True, includes time as additional input dimension
        '''
        super().__init__()
        self.time_varying = time_varying
        dim = arch[0]
        out_dim = dim
        w = arch[1]
        num_layers = arch[2]

        # Default to ReLU if no activation specified
        if len(arch) == 3:
            activation_fn = torch.nn.ReLU()
        else:
            activation_fn = arch[3]

        # Build network architecture
        layers = []
        # Input layer (add extra dim if time-varying)
        layers.append(torch.nn.Linear(dim + (1 if time_varying else 0), w))
        layers.append(torch.nn.LayerNorm(w))
        layers.append(activation_fn)
        # Hidden layers
        for i in range(num_layers-2):
            layers.append(torch.nn.Linear(w, w))
            layers.append(torch.nn.LayerNorm(w))
            layers.append(activation_fn)
        # Output layer
        layers.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        '''
        Forward pass through network.
        Handles time-varying case by excluding last dimension if not time_varying.
        '''
        if not self.time_varying:
            return self.net(x[:,:-1])
        return self.net(x)


    
class torch_wrapper(torch.nn.Module):
    """
    Wraps model to torchdyn compatible format by handling time input.
    Concatenates time dimension with input for time-varying dynamics.
    """
    # From torchcfm module

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # Expand scalar time to match batch dimension
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    


class ParameterizedMLP(nn.Module):
    def __init__(self, arch, time_varying=True):
        '''
        MLP where weights and biases are provided as a flat parameter vector.
        Useful for optimization over network parameters.
        
        Input:
            arch: [dim, width, num_layers]
                - dim: Input/output dimension
                - width: Hidden layer width
                - num_layers: Total number of layers
            time_varying: If True, includes time as additional input dimension
        '''
        super().__init__()
        self.dim = arch[0]
        self.w = arch[1]
        self.num_layers = arch[2]
        if len(arch) == 3:
            self.activation_fn = F.relu
        else:
            self.activation_fn = arch[3]
        self.time_varying = time_varying
        

    def forward(self, x, theta):
        '''
        Forward pass using parameters from theta vector.
        
        Input:
            x: Input tensor
            theta: Flat vector containing all weights and biases
        '''
        current_idx = 0 
        h = x 
        dim = self.dim + (1 if self.time_varying else 0)

        # Input layer parameters
        w_size = (dim)*self.w
        b_size = self.w
        w = theta[current_idx:current_idx+w_size].view(self.w,dim)
        current_idx += w_size
        b = theta[current_idx:current_idx+b_size]
        current_idx += b_size
        h = F.linear(h,w,b)
        h = F.layer_norm(h, [h.size(-1)])
        h = self.activation_fn(h)

        # Hidden layer
        # Hidden layers share width w*w for mat and w for bias
        w_size = self.w*self.w
        for _ in range(self.num_layers-2):
            w = theta[current_idx:current_idx+w_size].view(self.w,self.w)
            current_idx += w_size
            b = theta[current_idx:current_idx + b_size]
            current_idx += b_size
            h = F.linear(h,w,b)
            h = F.layer_norm(h, [h.size(-1)])
            h = self.activation_fn(h)

        # Output layer parameters
        w_size = self.w*self.dim
        b_size = self.dim
        w = theta[current_idx:current_idx+w_size].view(self.dim,self.w)
        current_idx += w_size
        b = theta[current_idx:current_idx+b_size]
        h = F.linear(h,w,b)

        return h


class ParameterizedWrapper(nn.Module):
    '''
    Wrapper for parameterized models that handles time input and parameter management.
    Includes dummy parameter to satisfy torchdyn requirements.
    '''
    def __init__(self, model, theta):
        super().__init__()
        self.model = model
        self.theta = theta
        # Dummy parameter to avoid message from torchdyn library
        self.register_parameter('theta_', nn.Parameter(torch.tensor(0.0)))
        
        

    def forward(self, t, x, *args, **kwargs):
        '''
        Forward pass that handles time dimension and parameter passing.
        Ensures time tensor matches batch dimension of input.
        '''
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])[:,None]
        assert t.shape[0] == x.shape[0]  # Ensure matching batch sizes
        return self.model(torch.cat([x,t],1), self.theta)

def order_state_to_tensor(order_state):
    '''
    Converts a dictionary of state tensors into a single flat tensor.
    Useful for parameter optimization and state management.
    
    Input:
        order_state: Dictionary of state tensors
    Output:
        torch.tensor: Flattened and concatenated tensor of all states
    '''
    out = []
    for value in list(order_state.values()):
        out.append(value.flatten())
    return torch.cat(out)