import torch as torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from 


# MLP model
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(torch.nn.Module):
    def __init__(self, arch, time_varying=True):
        '''
        Input:
            arch:  [dim,width,num_layers,activation_fn]
            time_varying: Boolean
        Ouput:
            nn.Module
        '''
        super().__init__()
        self.time_varying = time_varying
        dim = arch[0]
        out_dim = dim
        w = arch[1]
        num_layers = arch[2]

        if len(arch) == 3:
            activation_fn = torch.nn.ReLU()
        else:
            activation_fn = arch[3]

        layers = []
        layers.append(torch.nn.Linear(dim + (1 if time_varying else 0), w))
        layers.append(activation_fn)
        for i in range(num_layers-2):
            layers.append(torch.nn.Linear(w, w))
            layers.append(activation_fn)

        layers.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        if not self.time_varying:
            return self.net(x[:,:-1])
        return self.net(x)


    
class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    # From torchcfm module

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    


class ParameterizedMLP(nn.Module):
    def __init__(self,arch,time_varying = True):
        '''
        Input:
            arch = [dim,width,num_layers]
            time_varying: Boolean
        Output:
            nn.Module
        
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
        

    def forward(self,x,theta):
        # assign theta manually 
        current_idx = 0 
        h = x 

        dim =  self.dim +( 1 if self.time_varying else 0)

        # Input layer
        w_size = (dim)*self.w
        b_size = self.w
        w = theta[current_idx:current_idx+w_size].view(self.w,dim)
        current_idx += w_size
        b = theta[current_idx:current_idx+b_size]
        current_idx += b_size
        h = F.linear(h,w,b)
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
            h = self.activation_fn(h)

        # Output layer

        w_size = self.w*self.dim
        b_size = self.dim
        w = theta[current_idx:current_idx+w_size].view(self.dim,self.w)
        current_idx += w_size
        b = theta[current_idx:current_idx+b_size]
        h = F.linear(h,w,b)

        return h


class ParameterizedWrapper(nn.Module):

    def __init__(self,model,theta):
        super().__init__()
        self.model = model
        self.theta = theta
        # Dummy parameter to avoid message from torchdyn library
        self.register_parameter('theta_', nn.Parameter(torch.tensor(0.0)))
        # self.register_parameter('theta_', nn.Parameter(theta, requires_grad=True))
        

    def forward(self,t,x,*args,**kwargs):
        # Check if t is a scalar, used in torchdyn
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])[:,None]
        # return self.model(torch.cat([x,t.repeat(x.shape[0])[:,None]],1),self.theta)
        assert t.shape[0] == x.shape[0] # Check if t and x have the same batch size
        # print('t',t.shape,'x',x.shape)
        return self.model(torch.cat([x,t],1),self.theta)

def order_state_to_tensor(order_state):
    '''
    Input:
        order_state: dict
    Output:
        torch.tensor
    '''
    out = []
    for value in list(order_state.values()):
        out.append(value.flatten())

    return torch.cat(out)