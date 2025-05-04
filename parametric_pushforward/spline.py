'''
Neural Transport Map Splines for Density Path Optimization

This module implements a method for finding optimal paths between probability distributions
using neural transport maps and spline interpolation. The approach combines ideas from
optimal transport, neural ODEs, and path optimization.

Core Concepts:
-------------
1. Neural Transport Maps:
   - Neural networks that transform samples from one distribution to another
   - Parameters of these networks define the transport map
   - Uses Neural ODEs to generate continuous flows

2. Spline Interpolation:
   - Creates smooth paths through the parameter space of neural networks
   - Interpolates between parameters of transport maps at endpoints
   - Supports both linear and cubic spline interpolation

3. Energy Functions:
   - Kinetic Energy: Measures the "speed" of distribution evolution
   - Potential Energy: Optional costs for path constraints
   - Entropy: Tracks density evolution along the path
   - Fisher Information: Geometric structure of the path

4. Optimization:
   - Path Optimization: Finds optimal interior points of the spline
   - Coupling Optimization: Matches endpoint distributions
   - Geodesic Warmup: Initializes path using Wasserstein geometry

Key Components:
-------------
- Spline Class: Core implementation of the interpolation and optimization
- Assemble_spline: Factory function to create spline instances
- Energy Computations: Methods for computing various path costs
- Sampling: Functions to generate samples along the path

Usage:
------
# Create a spline between two distributions
spline, t = Assemble_spline(
    theta0=initial_params,
    theta1=target_params,
    arch=[input_dim, hidden_dim, num_layers, activation],
    data0="gaussian",  # Initial distribution
    data1="mixture",   # Target distribution
)

# Optimize the path
spline.optimize_path(...)

# Generate samples along the path
samples = spline.gen_sample_trajectory(...)

References:
----------
[1] Neural Transport Maps for Density Path Optimization
[2] Optimal Transport and Wasserstein Distance
[3] Neural Ordinary Differential Equations

Authors: [Your names]
License: [License info]
'''

# Add project root to path for imports
from pathlib import Path
project_root = Path(__file__).parent.absolute()
import sys
sys.path.append(str(project_root))

import torch as torch

from interpolation import cubic_interp,dervi_cubic_interp

from parametric_mlp import ParameterizedMLP,ParameterizedWrapper
from odes_log_score import trajectory_log_likelihood_exact_rhs,trajectory_score_exact_rhs,trajectory_log_likelihood_score_exact_rhs


# parametric_oode_solvers modifies the definitions of the step integrators in odeint
# ALWAYS CALL IT BEFORE.
import parametric_ode_solvers
from torchdyn.core import NeuralODE
from torchdyn.numerics import odeint

from tqdm import tqdm 
# import ot as pot

import data_sets as toy_data

import matplotlib.pyplot as plt


def Assemble_spline(theta0,theta1,arch,data0 = None,data1 = None,ke_modifier= None,potential = [None],number_of_knots = 3,spline = 'cubic',device = 'cpu',prior_dist = None,p=2):
    '''
    Creates a spline interpolation between two sets of neural network parameters.

    Args:
        theta0: Initial parameter vector
        theta1: Final parameter vector  
        arch: Network architecture [input_dim, hidden_dim, num_layers, activation]
        data0: Initial distribution identifier (must match toy_data dataset)
        data1: Final distribution identifier (must match toy_data dataset)
        ke_modifier: Optional functions to modify kinetic energy
        potential: List of potential energy terms
        number_of_knots: Number of interior interpolation points
        spline: Interpolation type ('cubic' or 'linear')
        device: Computation device
        prior_dist: Base distribution (defaults to N(0,I))
        p: Norm used for kinetic energy

    Returns:
        spline: Initialized Spline object
        t: Time points for interpolation
    '''
    
    total_knots = number_of_knots + 2 # Inlcude bds
    t = torch.linspace(0,1,total_knots,device = device)

    tt = t.view(-1,1)

    # Initialize spline

    thetat = (theta0*(1-tt)**(1) + theta1*tt**(1) ).unsqueeze(0)# #(2*tt**3 - 3*tt**2 + 1)*theta0 + (-2*tt**3 + 3*tt**2)*theta1

    return Spline(t,thetat,arch,data0 = data0,data1 = data1,ke_modifier=ke_modifier,potential = potential,spline_type = spline,prior_dist  = prior_dist,p=p),t


class Spline(torch.nn.Module):

    def __init__(self,t,xt,arch,data0 = None,data1 = None,ke_modifier= None,potential = None,spline_type = "linear",prior_dist = None,p=2):
        '''
        t : (T,)
        xt: (1,T,D)
        arch: [input_dim,hidden_dim,num_layers]
        data0: string, needs to be compatible with a data set defined in toy_data
        data1: string, needs to be compatible with a data set defined in toy_data
        ke_modifier: list of functions to linearly modify the kinetic energy or None. Functions R^d->R^d
        potential: list of functions with maybe two strings : 'entropy' and 'fisher_information'. Functions: R^d->R
        spline_type: str: 'linear' or 'cubic'
        prior_dist: torch.distributions.Distribution if None, then MultivariateNormal(0,I) is used
        p: int, norm for kinetic energy
        trace_est: str, 'hutchinson' or 'exact'
        '''
        super(Spline,self).__init__()
        _,T,D = xt.shape
        assert t.shape ==(T,) and T>2, 'Need more than 2 time points'
        assert t.device == xt.device, 'Time points and data points should be on the same device'

        t = t.detach().clone()
        xt = xt.permute(1,0,2).detach().clone() #(T,1,D)
        self.data0 = data0
        self.data1 = data1
        self.sample_dim = arch[0]
        if prior_dist is not None:
            self.prior_dist = prior_dist
        else:
            self.prior_dist = torch.distributions.MultivariateNormal(torch.zeros(self.sample_dim),torch.eye(self.sample_dim))#.to(t.device)
        self.T = T
        self.D = D
        self.device = t.device
        self.spline_type = spline_type
        self.arch = arch
        self.non_linear_pot = False
        self.entropy_pot = False
        self.fisher_pot = False
        self.p = p
        self.sigma = 1
        # Update variables to deal with entropy
        if 'entropy' in potential:
            self.entropy_pot = True
            self.non_linear_pot = True
            potential.remove('entropy')
        # Update variables to deal with fisher information
        if 'fisher_information' in potential:
            self.fisher_pot = True
            self.non_linear_pot = True

            potential.remove('fisher_information')
        
        # Setup potential energy functions
        if potential[0] is not None:
            self.potential = potential
        else:
            self.potential = None
        # Setup kinetic energy functions
        self.ke_modifier = ke_modifier
        self.dt_coupling = torch.tensor([0.05]).to(self.device)

        # Register parameters
        self.register_buffer('t',t)
        self.register_buffer('t_epd',t.reshape(-1,1).expand(-1,1))
        self.register_parameter('x0',torch.nn.Parameter(xt[0].reshape(1,1,D))) #.reshape(1,1,D)
        self.register_parameter('knots',torch.nn.Parameter(xt[1:-1]))
        self.register_parameter('x1',torch.nn.Parameter(xt[-1].reshape(1,1,D)))
        

    @property
    def xt(self):
        return torch.cat([self.x0,self.knots,self.x1],dim = 0).permuter(1,0,2)

    def interp(self,query_t):
        '''
        query_t:(S,)-> yt:(1,S,D)
        '''

        (S,) = query_t.shape
        query_t = query_t.reshape(-1,1).expand(-1,1)

        xt = torch.cat([self.x0,self.knots,self.x1],dim = 0)
        yt = cubic_interp(self.t_epd,xt,query_t)
        yt = yt.permute(1,0,2)  #(1,S,D)

        return yt

    def deriv(self,query_t):
        '''
        query_t:(S,)-> yt:(1,S,D)
        '''

        (S,) = query_t.shape
        query_t = query_t.reshape(-1,1).expand(-1,1)

        xt = torch.cat([self.x0,self.knots,self.x1],dim = 0)
        yt_p =  dervi_cubic_interp(self.t_epd,xt,query_t)

        return yt_p.permute(1,0,2)

    def forward(self,t):
        '''
        t:(S,)-> yt:(1,S,D)
        '''
        return self.interp(t)

    def gen_sample_trajectory(self, x0=None, num_samples=1000, t_traj=torch.linspace(0,1,10),
                             time_steps_node=10, solver='euler', sensitivity='adjoint', forward=True):
        '''
        Generates samples along the interpolated path by pushing forward samples through 
        the sequence of transport maps.

        The method handles several cases:
        1. Basic transport without additional terms
        2. Transport with entropy tracking
        3. Transport with Fisher information
        4. Transport with both entropy and Fisher information

        Args:
            x0: Optional initial samples (if None, samples from prior_dist)
            num_samples: Number of samples to generate
            t_traj: Time points to evaluate samples at
            time_steps_node: Number of integration steps for NODE solver
            solver: ODE solver type ('euler' or 'midpoint')
            sensitivity: Gradient computation method ('adjoint' or 'autograd')
            forward: Direction of integration

        Returns:
            Depending on configuration:
            - samples_path only
            - (log_density_path, samples_path)
            - (norm_score_path, samples_path)
            - (log_density_path, norm_score_path, samples_path)
        '''
        # Sampes to pushforward
        if x0 is None :
            z = self.prior_dist.sample((num_samples,)).to(self.device)
        else:
            z = x0.clone()
        # Initial condition for entropy
        if self.entropy_pot and not self.fisher_pot:
            z = torch.cat([self.prior_dist.log_prob(z)[:,None],z],dim = -1)
        # Initial condition for the fisher information potential
        if self.fisher_pot and not self.entropy_pot:
            # The initial condition for the fisher information potential is -z 
            z = torch.cat([-z,z],dim = -1)
        # Initial condition for entropy and fisher information 
        if self.fisher_pot and self.entropy_pot:
            z = torch.cat([self.prior_dist.log_prob(z)[:,None],-z,z],dim = -1)

        t_traj = t_traj.to(self.device)
        time_steps_traj = t_traj.shape[0]

        # Output points
        samples_path = torch.zeros(num_samples,time_steps_traj,self.sample_dim,device=self.device)

        if self.entropy_pot:
            log_density_path = torch.zeros(time_steps_traj,device=self.device)

        if self.fisher_pot:
            norm_score_path = torch.zeros(time_steps_traj,device=self.device)

        # Time points
        t_node = torch.linspace(0,1,time_steps_node,device=self.device)

        # Build weight interpolation 
        theta_t = self.interp(t_traj)[0]

        # Sample trajecotry
        for i in range(time_steps_traj):
            # Obtain weights
            theta = theta_t[i]
            # Build model
            model = ParameterizedWrapper(ParameterizedMLP(self.arch,time_varying=True).to(self.device),theta)
            # Setup NODE
            node_theta = NeuralODE(model,solver=solver,sensitivity=sensitivity,atol = 1e-4,rtol = 1e-4).to(self.device)
            # Obtain samples
            if self.non_linear_pot == False:
                samples = node_theta.trajectory(z.clone(),t_span = t_node)
                samples_path[:,i,:] = samples[-1,:,:]
            # Samples and log density, no fisher information
            elif self.entropy_pot == True and self.fisher_pot == False:
                args_local = [node_theta]
                _,log_samples = odeint(trajectory_log_likelihood_exact_rhs,z.clone(),t_node,solver = solver,args = args_local)
                log_density,samples = log_samples[-1,:,0],log_samples[-1,:,1:]
                samples_path[:,i,:] = samples
                log_density_path[i] = log_density.mean()
            # Samples and fisher information, no log density
            elif self.fisher_pot == True and self.entropy_pot == False:
                args_local = [node_theta,self.sample_dim]
                _,score_samples = odeint(trajectory_score_exact_rhs,z.clone(),t_node,solver = solver,args = args_local)
                score,samples = score_samples[-1,:,:self.sample_dim],score_samples[-1,:,self.sample_dim:]
                samples_path[:,i,:] = samples
                norm_score_path[i] = torch.mean(torch.norm(score,dim = -1)**2)
            # Samples, fisher info and log density
            elif self.fisher_pot == True and self.entropy_pot == True:
                args_local = [node_theta,self.sample_dim]
            
                _,log_score_samples = odeint(trajectory_log_likelihood_score_exact_rhs,z.clone(),t_node,solver = solver,args = args_local)
                log_density,score,samples = log_score_samples[-1,:,0],log_score_samples[-1,:,1:self.sample_dim+1],log_score_samples[-1,:,self.sample_dim+1:]
                samples_path[:,i,:] = samples
                log_density_path[i] = log_density.mean()
                norm_score_path[i] = torch.mean(torch.norm(score,dim = -1)**2)

        if self.entropy_pot and not self.fisher_pot:
            return log_density_path,samples_path
        elif self.fisher_pot and not self.entropy_pot:
            return norm_score_path,samples_path
        elif self.fisher_pot and self.entropy_pot:
            return log_density_path,norm_score_path,samples_path
        return samples_path
    
    def potential_energy(self, samples_path):
        '''
        Computes the potential energy along a path of samples.

        Args:
            samples_path: (batch_size, num_timesteps, dim) tensor of sample trajectories

        Returns:
            potential_energy: (num_timesteps,) tensor containing potential energy at each timestep,
                            averaged over batch
        '''
        
        if self.potential is not None:
            potential_energy = self.potential[0](samples_path)
            for i in range(1,len(self.potential)):
                potential_energy += self.potential[i](samples_path)
            # potential_energy = self.potential(samples_path)
            potential_energy = potential_energy.mean(dim = 0)
        else:
            potential_energy = torch.zeros(samples_path.shape[1],device = samples_path.device)
        return potential_energy
    
    def kinetic_energy(self, samples_path, times_path):
        '''
        Computes the kinetic energy along a path of samples using finite differences.
        
        Args:
            samples_path: (batch_size, num_timesteps, dim) tensor of sample trajectories
            times_path: (num_timesteps,) tensor of timepoints

        Returns:
            ke: (num_timesteps,) tensor containing kinetic energy at each timestep,
                computed as p-norm of velocities averaged over batch
        '''
        p = self.p
        # Compute forward difference
        difference = (samples_path[:,1:,:] - samples_path[:,:-1,:])
        dt = (times_path[1:] - times_path[:-1] + 1e-6).unsqueeze(-1)

        # Compute velocity
        velocity = difference/dt

        # Compute centered difference for interior points
        m = (velocity[:,1:,:] + velocity[:,:-1,:])/2
        # Velocity estimate for trajectory
        m = torch.cat([velocity[:,:1,:],m,velocity[:,-1:,:]],dim = 1)
        if self.ke_modifier is not None:
            modified_energy = torch.zeros_like(m)
            for f in self.ke_modifier:
                # Samples_path are permuted to (Bs,s,D)
                
                modified_energy += f(samples_path,times_path)
            # Permute back to (Bs,D,s)
            modified_energy = modified_energy
            m += modified_energy
        
        # Compute kinetic energy
        ke = m.norm(p = p, dim = -1)**p
        ke = ke.mean(dim = 0) # mean over samples at each time step

        return ke


    def acceleration_energy(self, samples_path, times_path):
        '''
        Computes the acceleration energy (second derivative) along a path using finite differences.
        
        Args:
            samples_path: (batch_size, num_timesteps, dim) tensor of sample trajectories
            times_path: (num_timesteps,) tensor of timepoints

        Returns:
            ae: (num_timesteps,) tensor containing acceleration energy at each timestep,
                computed as L2 norm of accelerations averaged over batch
        '''
        # Compute forward difference
        difference = (samples_path[:,1:,:] - samples_path[:,:-1,:])
        dt = (times_path[1:] - times_path[:-1] + 1e-6).unsqueeze(-1)

        # Compute velocity
        velocity = difference/dt

        # Compute centered difference for interior points
        m = (velocity[:,1:,:] + velocity[:,:-1,:])/2
        # Velocity estimate for trajectory
        m = torch.cat([velocity[:,:1,:],m,velocity[:,-1:,:]],dim = 1)

        # Compute forward difference
        difference = (m[:,1:,:] - m[:,:-1,:])
        dt = (times_path[1:] - times_path[:-1] + 1e-6).unsqueeze(-1)

        # Compute acceleration
        acceleration = difference/dt

        # Compute centered difference for interior points
        m = (acceleration[:,1:,:] + acceleration[:,:-1,:])/2
        # Velocity estimate for trajectory
        m = torch.cat([acceleration[:,:1,:],m,acceleration[:,-1:,:]],dim = 1)

        # Compute acceleration
        ae = m.norm(p = 2, dim = -1)**2
        ae = ae.mean(dim = 0) # mean over samples at each time step

        return ae

    def lagrangian(self,samples_path,times_path,log_density = None,score = None):
        '''
        Input:
            samples:(Bs,D,s)
            times_path:(s,) torch.tensor
        Output:
            lagrangian:(s,)
        '''
        # ke = self.kinetic_energy(samples_path,times_path)

        ke = self.kinetic_energy(samples_path,times_path)
        # ke = self.acceleration_energy(samples_path,times_path)
        ke = torch.trapz(ke,times_path)/2#torch.sqrt(torch.trapz(ke,times_path))/2#
        pe = self.potential_energy(samples_path)
        if log_density is not None:
            pe = pe + log_density*50
        if score is not None:
            pe = pe + score*(self.sigma**4/8)
        pe = torch.trapz(pe,times_path)

        return ke + pe,ke,pe

    def optimize_path(self, epochs, optimizer, scheduler, t_partition, ema=None, t_node=10, bs=1000, x0=None):
        '''
        Optimizes the interior points of the spline path while keeping endpoints fixed.
        
        Args:
            epochs: Number of optimization iterations
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            t_partition: Number of time points to evaluate trajectory
            ema: Exponential moving average for parameters (optional)
            t_node: Number of integration steps for NODE solver
            bs: Batch size for sampling
            x0: Optional fixed initial samples, if None samples from prior

        Returns:
            outputs: Dictionary containing optimization history:
                    - 'lagrangian': Total cost at each epoch
                    - 'kinetic': Kinetic energy at each epoch
                    - 'potential': Potential energy at each epoch
        '''

        
        pbar = tqdm(range(epochs))
        outputs = {'lagrangian':[],'kinetic':[],'potential':[]}
        
        t_traj = torch.linspace(0,1,t_partition).to(self.device)        
        if x0 is not None:
            x0_eval = x0.clone()
        t_traj = torch.linspace(0,1,t_partition).to(self.device)
        x1_clone = self.x1.clone()
        x0_clone = self.x0.clone()
        for _ in pbar:
            
            if x0 is None:
                x0_eval = self.prior_dist.sample((bs,)).to(self.device)

            optimizer.zero_grad()

            if self.entropy_pot and not self.fisher_pot:
                log_density,samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,ke,pe = self.lagrangian(samples_path,t_traj,log_density = log_density)
                # hamiltonian = self.hamiltonian(samples_path,t_traj,log_density = log_density)
            elif self.fisher_pot and not self.entropy_pot:
                norm_score,samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,ke,pe = self.lagrangian(samples_path,t_traj,score = norm_score)
                # hamiltonian = self.hamiltonian(samples_path,t_traj,score = norm_score)
            elif self.fisher_pot and self.entropy_pot:
                log_density,norm_score,samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,ke,pe = self.lagrangian(samples_path,t_traj,log_density = log_density,score = norm_score)
                # hamiltonian = self.hamiltonian(samples_path,t_traj,log_density = log_density,score = norm_score)
            else:
                samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,ke,pe = self.lagrangian(samples_path,t_traj)
                # hamiltonian = self.hamiltonian(samples_path,t_traj)
            
            outputs['lagrangian'].append(lagrangian.detach().cpu().numpy())
            outputs['kinetic'].append(ke.detach().cpu().numpy())
            outputs['potential'].append(pe.detach().cpu().numpy())
                       

            total_cost = lagrangian
            (total_cost).backward()

            
            optimizer.step()
            
            ema.update()
            
            self.x0.data = x0_clone.data
            self.x1.data = x1_clone.data
                                        
                
            # torch.nn.utils.clip_grad_norm_(self.parameters(), .01)

            pbar.set_description(f'Path_opt: {lagrangian.item()},ke:{ke.item()},pe:{pe.item()}',refresh=True)    
        scheduler.step()
        return outputs
    
    def optimize_coupling(self, epochs, optimizer, scheduler, t_partition, ema=None, t_node=10, 
                         bs=1000, weight_bd=1000, x0=None):

        '''
        Optimizes the endpoint parameters of the spline to match target distributions.
        
        Args:
            epochs: Number of optimization iterations
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            t_partition: Number of time points to evaluate trajectory
            ema: Exponential moving average for parameters (optional)
            t_node: Number of integration steps for NODE solver
            bs: Batch size for sampling
            weight_bd: Weight for boundary matching terms
            x0: Optional fixed initial samples, if None samples from prior

        Returns:
            outputs: Dictionary containing optimization history:
                    - 'bd_0': Initial boundary cost at each epoch
                    - 'bd_1': Final boundary cost at each epoch
        '''

        
        pbar = tqdm(range(epochs))
        outputs = {'bd_0':[],'bd_1':[]}

        
        t_traj = torch.linspace(0,1,t_partition).to(self.device)        
        if x0 is not None:
            x0_eval = x0.clone()
        xt_clone = self.knots.clone()
        for _ in pbar:
            
            if x0 is None:
                x0_eval = self.prior_dist.sample((bs,)).to(self.device)

            optimizer.zero_grad()
            
            if self.entropy_pot and not self.fisher_pot:
                log_density,samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,_,_ = self.lagrangian(samples_path,t_traj,log_density = log_density)
                # hamiltonian = self.hamiltonian(samples_path,t_traj,log_density = log_density)
            elif self.fisher_pot and not self.entropy_pot:
                norm_score,samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,_,_ = self.lagrangian(samples_path,t_traj,score = norm_score)
                # hamiltonian = self.hamiltonian(samples_path,t_traj,score = norm_score)
            elif self.fisher_pot and self.entropy_pot:
                log_density,norm_score,samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,_,_ = self.lagrangian(samples_path,t_traj,log_density = log_density,score = norm_score)
                # hamiltonian = self.hamiltonian(samples_path,t_traj,log_density = log_density,score = norm_score)
            else:
                samples_path = self.gen_sample_trajectory(x0 = x0_eval,num_samples=bs,t_traj = t_traj,time_steps_node=t_node,solver = 'midpoint',sensitivity='autograd')
                lagrangian,_,_ = self.lagrangian(samples_path,t_traj)
                # hamiltonian = self.hamiltonian(samples_path,t_traj)
            
            
            terminal_cost0 = self.terminal_cost(boundary = 0,batch_size=2*bs,weight_terminal=weight_bd)
            terminal_cost1 = self.terminal_cost(boundary = 1,batch_size=2*bs,weight_terminal=weight_bd)
            
            outputs['bd_0'].append(terminal_cost0.detach().cpu().numpy())
            outputs['bd_1'].append(terminal_cost1.detach().cpu().numpy())
                
    
            terminal_cost = terminal_cost0 + terminal_cost1
            total_cost =  terminal_cost + lagrangian     
            
            (total_cost).backward()
    
            optimizer.step()
            
            ema.update()
            
            self.knots.data = xt_clone.data
                            
                
            # torch.nn.utils.clip_grad_norm_(self.parameters(), .01)

            pbar.set_description(f'Bd_0: {terminal_cost0.item()},bd_1:{terminal_cost1.item()},lagrangian:{lagrangian.item()}',refresh=True)    
        scheduler.step()

        return outputs
    
    def terminal_cost(self, boundary=0, batch_size=1000, weight_terminal=1):
        '''
        Computes cost for matching endpoint distributions.
        
        Args:
            boundary: 0 for initial distribution, 1 for final distribution
            batch_size: Number of samples for estimation
            weight_terminal: Weight for the terminal cost term

        Returns:
            loss: Weighted MSE between target and predicted velocities at boundary
        '''
        if boundary == 0:
            data = self.data0
            x = self.x0.flatten()
        else:
            data = self.data1
            x = self.x1.flatten()

        data_set = torch.tensor(toy_data.inf_train_gen(data,batch_size=batch_size,dim = self.sample_dim)).float().to(self.device)
        z = self.prior_dist.sample((batch_size,)).to(self.device)   
        t = torch.rand(batch_size).to(self.device)
        
        xt = data_set*t[:,None] + (1-t[:,None])*z
        ut = data_set-z
        
        # Compute the model prediction
        model = ParameterizedWrapper(ParameterizedMLP(self.arch,time_varying=True).to(self.device),x)
        vt = model(t[:,None],xt)
        loss = torch.mean(torch.sum((ut - vt)**2,dim = 1))*weight_terminal

        return loss
    
    
    
    def hamiltonian(self,samples_path,times_path,log_density = None,score = None):
        '''
        Input:
            samples:(Bs,D,s)
            times_path:(s,) torch.tensor
        Output:
            hamiltonian:(s,)
        '''
        ke = self.kinetic_energy(samples_path,times_path)
        pe = self.potential_energy(samples_path)
        if log_density is not None:
            pe = pe + log_density
        if score is not None:
            pe = pe + score*(self.sigma**4/8)
        hamiltonian = ke - pe
        return hamiltonian

    def push_forward(self, theta, z, t_node=10):
        '''
        Pushes samples forward through a neural ODE with given parameters.
        
        Args:
            theta: Parameters for the neural ODE
            z: (batch_size, dim) tensor of input samples
            t_node: Number of integration steps

        Returns:
            Transformed samples after flowing through the ODE
        '''
        # define parametric model
        model = ParameterizedWrapper(ParameterizedMLP(self.arch,time_varying=True).to(self.device),theta)
        # define neural ODE
        node = NeuralODE(model,solver='midpoint',sensitivity='autograd',atol = 1e-4,rtol = 1e-4).to(self.device)
        # solve neural ODE
        z = node.trajectory(z,t_span = torch.linspace(0,1,t_node).to(self.device))[-1]      
        return z


    def pull_back(self, theta, x, t_node=10):
        '''
        Pulls samples backward through a neural ODE with given parameters.
        
        Args:
            theta: Parameters for the neural ODE
            x: (batch_size, dim) tensor of input samples
            t_node: Number of integration steps

        Returns:
            Original samples obtained by running ODE backward
        '''
        # define parametric model

        model = ParameterizedWrapper(ParameterizedMLP(self.arch,time_varying=True).to(self.device),theta)
        # define neural ODE
        node = NeuralODE(model,solver='midpoint',sensitivity='autograd',atol = 1e-4,rtol = 1e-4).to(self.device)
        # solve neural ODE
        z = node.trajectory(x,t_span = torch.linspace(1,0,t_node).to(self.device))[-1]      

        return z

    def geodesic_warmup(self, optimizer, num_epochs=100):
        '''
        Initializes control points by optimizing for geodesic path in Wasserstein space.
        
        Args:
            optimizer: PyTorch optimizer
            num_epochs: Number of warmup iterations

        The method:
        1. Samples from initial distribution
        2. Pulls samples back to base space
        3. Optimizes path by minimizing kinetic energy of pushed forward samples
        '''
        num_samples = 1000
        fix_t_partition = self.T
        time_steps_node = 10
        
        t_traj = torch.linspace(0,1,fix_t_partition).to(self.device)

        t_traj = t_traj.to(self.device)
        time_steps_traj = t_traj.shape[0]
                

        for _ in range(num_epochs):
            x0 = torch.from_numpy(toy_data.inf_train_gen(self.data0,batch_size=num_samples,dim = self.sample_dim)).float().to(self.device)
            z = self.pull_back(self.x0.flatten(),x0)
            
            # Output points
            samples_path = torch.zeros(num_samples,time_steps_traj,self.sample_dim,device=self.device)
            
            # Build weight interpolation 
            theta_t = self.interp(t_traj)[0]
            # Sample trajecotry
            optimizer.zero_grad()
            for i in range(time_steps_traj):

                theta = theta_t[i]
                samples = self.push_forward(theta,z.clone(),t_node = time_steps_node)
                samples_path[:,i,:] = samples

            # Compute ke of samples
            ke = self.kinetic_energy(samples_path,t_traj)
            ke = torch.trapz(ke,t_traj)/2
            ke.backward()    
            optimizer.step()

    

    