'''
https://github.com/SebasGutHdz/PDPO/tree/main/

This file contains the right-hand side (RHS) equations for ODEs that compute:
1. The log density (log likelihood)
2. The score function (gradient of log density)
These are used in the parametric pushforward optimization framework.
'''
import torch

def div_vector(x_in, x_out, **kwargs):
    '''
    Computes the divergence of a vector field using automatic differentiation.
    The divergence is the sum of partial derivatives ∂v_i/∂x_i.
    
    Input:
        x_out: [Bs,d] - Output of vector field v(x)
        x_in: [Bs,d] - Input points x where Bs=batch size, d=dimension
    Output:
        divergence [Bs] - Scalar divergence at each input point
    '''

    jacobian_mat_diag = torch.zeros(x_in.shape[0], x_in.shape[1]).to(x_in.device)
    for i in range(x_out.shape[1]):
        grad_out = torch.autograd.grad(x_out[:, i].sum(), x_in, create_graph=True, retain_graph=True)[0]
        jacobian_mat_diag[:,i] = grad_out[:,i]
    
    div = jacobian_mat_diag.sum(dim = 1)

    return div


def trajectory_log_likelihood_exact_rhs(t, x, *args):
    '''
    Computes the RHS of the ODE for evolving the log likelihood:
    ∂log(ρ)/∂t = -div(v)  
    
    Input:
        t: [1] tensor - Current time
        x: [Bs,d+1] tensor - First entry is log likelihood, last d entries are samples
        args[0]: Neural ODE model
    Output:
        [Bs,d+1] tensor - RHS of the ODE for [log_likelihood, sample_trajectory]
    '''
    node = args[0]
    x_in = x[:, 1:].requires_grad_(True).to(node.device)#.clone().detach()
    # Compute dynamics and divergence vector field
    x_out = node.vf(t, x_in)
    nabla_dot_v = div_vector(x_in, x_out)   
    
    return torch.cat([-nabla_dot_v[:,None],x_out],dim = 1)

def jac_vector(x_in, x_out, **kwargs):
    '''
    Computes the Jacobian matrix of a vector field using automatic differentiation.
    The Jacobian contains all partial derivatives ∂v_i/∂x_j.
    
    Input:
        x_out: [Bs,d] - Output of vector field v(x)
        x_in: [Bs,d] - Input points x
    Output:
        jacobian: [Bs,d,d] - Jacobian matrix at each input point
    '''

    jacobian_mat_diag = torch.zeros(x_in.shape[0], x_in.shape[1],x_in.shape[1]).to(x_in.device)
    for i in range(x_out.shape[1]):
        grad_out = torch.autograd.grad(x_out[:, i].sum(), x_in, create_graph=True, retain_graph=True,allow_unused=True)[0]
        jacobian_mat_diag[:,i,:] = grad_out#.T

    return jacobian_mat_diag

def trajectory_score_exact_rhs(t, x, *args):
    '''
    Computes the RHS of the ODE for evolving the score function:
    ∂(∇log(ρ))/∂t = -∇(div(v)) - (∇v)^T ∇log(ρ)
    
    Input:
        t: [1] tensor - Current time
        x: [Bs,2d] tensor - First d entries are score, last d entries are samples
        args: [node,d] - Neural ODE model and dimension
    Output:
        [Bs,2d] tensor - RHS of the ODE for [score, sample_trajectory]
    '''
    node = args[0]
    d = args[1]
    score_in = x[:, :d].clone().detach().requires_grad_(True).to(node.device) #[Bs,d]
    x_in = x[:, d:].clone().detach().requires_grad_(True).to(node.device) #[Bs,d]
    # Compute dynamics and divergence vector field
    x_out = node.vf(t, x_in)
    jacob_mat = jac_vector(x_in, x_out) #[Bs,d,d]
    nabla_dot_v = jacob_mat.sum(dim=1)
    grad_nabla_dot_v = torch.autograd.grad(nabla_dot_v.sum(), x_in, create_graph=True)[0]
    jac_vec_score = torch.bmm(jacob_mat, score_in[:, :, None])[:, :, 0] #[Bs,d]
    rhs_score = -grad_nabla_dot_v - jac_vec_score
    
    
    return torch.cat([rhs_score,x_out],dim = 1)

def trajectory_log_likelihood_score_exact_rhs(t, x, *args):
    '''
    Computes the coupled ODEs for both log density and score evolution:
    ∂log(ρ)/∂t = -div(v)
    ∂(∇log(ρ))/∂t = -∇(div(v)) - (∇v)^T ∇log(ρ)
    
    Input:
        t: [1] tensor - Current time
        x: [Bs,2d+1] tensor - Structure: [log_density, score (d dim), samples (d dim)]
        args: [node,d] - Neural ODE model and dimension
    Output:
        [Bs,2d+1] tensor - RHS of coupled ODEs for [log_density, score, sample_trajectory]
    '''
    node = args[0]
    d = args[1]
    
    # Extract score and position from input tensor
    score_in = x[:, 1:d+1].clone().detach().requires_grad_(True).to(node.device) #[Bs,d]
    x_in = x[:, d+1:].clone().detach().requires_grad_(True).to(node.device) #[Bs,d]
    
    # Compute vector field and its Jacobian
    x_out = node.vf(t, x_in)  # Evolution of samples
    jacob_mat = jac_vector(x_in, x_out) #[Bs,d,d]
    
    # Compute divergence and its gradient
    nabla_dot_v = jacob_mat.sum(dim=1)  # Divergence as sum of diagonal elements
    nabla_dot_v_r = torch.diagonal(jacob_mat, dim1=1, dim2=2).sum(dim=1)  # Alternative divergence computation
    grad_nabla_dot_v = torch.autograd.grad(nabla_dot_v.sum(), x_in, create_graph=True)[0]  # Gradient of divergence
    
    # Compute Jacobian-vector product for score evolution
    jac_vec_score = torch.bmm(jacob_mat, score_in[:, :, None])[:, :, 0] #[Bs,d]
    
    # RHS for score evolution
    rhs_score = -grad_nabla_dot_v - jac_vec_score
    
    # Combine all components: [log_density_evolution, score_evolution, sample_evolution]
    first_cat = torch.cat([-nabla_dot_v_r[:,None], rhs_score], dim=1)
    second_cat = torch.cat([first_cat, x_out], dim=1)
    return second_cat