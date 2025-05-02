'''



'''

from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
import sys
sys.path.append(str(project_root))
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

import parametric_pushforward.data_sets as toy_data
from reference_solutions.gaussian_solutions import monge_map
from parametric_pushforward.parametric_mlp import ParameterizedMLP,ParameterizedWrapper
from parametric_pushforward.opinion import est_directional_similarity,proj_pca

import parametric_pushforward.parametric_ode_solvers
from torchdyn.core import NeuralODE

from sklearn.decomposition import PCA
from sklearn.manifold import MDS,Isomap

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


from IPython.display import HTML


def display_bds(spline,n_points = 1000,device = 'cuda:0',time_steps= 10,z = None):
    '''
    Input:
        spline: torch.nn.Module, spline class
        n_points: int
        device: str

    Output:
        None
    '''
    d = spline.sample_dim
    z = spline.prior_dist.sample((n_points,)).to(device) if z is None else z
    # Boundary conditions
    theta0  = spline.x0[0][0]
    theta1 = spline.x1[0][0]
    
    z0 = spline.push_forward(theta0,z).detach().cpu()
    z1 = spline.push_forward(theta1,z).detach().cpu()

    plt.scatter(z0[:,0],z0[:,1],c = 'r',s =1,label = 'Initial')
    plt.scatter(z1[:,0],z1[:,1],c = 'b',s =1,label = 'Terminal')
    plt.legend()
    plt.show

def disimilarity_plot(x,ax)-> None:

    _,d = x.shape
    n_est = 5000
    directional_sim = est_directional_similarity(x,n_est)
    assert directional_sim.shape == (n_est,)

    directional_sim = directional_sim.cpu().detach().numpy()

    bins = 15

    _,_,patches = ax.hist(directional_sim,bins = bins,)

    colors = plt.cm.coolwarm(np.linspace(0,1,bins))

    for c,p in zip(colors,patches):
        plt.setp(p,'facecolor',c)

    ymax = 1000 if d == 2 else 2000
    ax.set_ylim(0,ymax)
    ax.set_xlim(0,1)

def disimilarity_snapshots(xt):
    '''
    xt: torch.tensor, shape (Bs,T,dim)
    '''
    _,T,dim = xt.shape

    idxs = np.linspace(0,T-1,4).astype(np.int32)

    fig,axs = plt.subplots(nrows = 2,ncols = 4,figsize = (20,10))
    ax = axs.ravel()
    fig.suptitle('Density Visualization and Directional Similarity Histogram',fontsize = 16)
    cmap = plt.cm.coolwarm
    for i in range(4):
        if dim > 2:
            # do pca 
            xt = proj_pca(xt)[0]

        ax[i].scatter(xt[:,idxs[i],0].cpu().detach().numpy(),xt[:,idxs[i],1].cpu().detach().numpy(),s = 1)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].set_xlim(-10,10)
        ax[i].set_ylim(-10,10)
        disimilarity_plot(xt[:,idxs[i],:],ax[i+4])
        ax[i].set_title('t = {:.3f}'.format(idxs[i]/(T-1)))

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    norm = plt.Normalize(-1, 1)  # Assuming similarity ranges from -1 to 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Add custom tick labels
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Highly Agree', 'Neutral', 'Highly Disagree'])
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)  # Adjusted bottom margin for colorbar
    plt.show()

    
def path_visualization_particles(xt,spline):
    '''
    Plot particles xt
    xt: torch.tensor, shape (Bs,T,dim)
    spline: torch.nn.Module, spline class. This is added to evaluate the meshgrid consistently across diff. examples
    '''
    from matplotlib.colors import LinearSegmentedColormap
    bs,T,dim = xt.shape
    #Only first two dimensions
    if dim > 2:
        xt = xt[:,:,:2]

    # # Plot upto 512 samples
    # if bs > 512:
    #     idx = np.random.choice(bs,512,replace = False)
    #     xt = xt[idx,:,:]
        # Plot
    # Custom colormap for time steps
    colors = plt.cm.autumn(np.linspace(0, 1, T))
    custom_cmap = LinearSegmentedColormap.from_list('autumn', colors)

    # Create visualization with potential function
    fig,ax = plt.subplots(figsize=(12, 10))

    lower_bound_x = xt[:, :, 0].min().item()
    upper_bound_x = xt[:, :, 0].max().item()
    lower_bound_y = xt[:, :, 1].min().item()
    upper_bound_y = xt[:, :, 1].max().item()

    # First plot the potential function
    X,Y = torch.meshgrid(  
        torch.linspace(lower_bound_x,upper_bound_x,100),
        torch.linspace(lower_bound_y,upper_bound_y,100),
        indexing='ij'  # Ensure consistent indexing
    )
    num_points = X.shape[0] * X.shape[1]
    # Create a full-dimensional input tensor but only populate first two dimensions
    full_dim_xy = torch.zeros(num_points,1,dim, device=spline.device)
    full_dim_xy[:,0, 0] = X.reshape(num_points).flatten()
    full_dim_xy[:,0, 1] = Y.reshape(num_points).flatten()

    

    if spline.potential is None:
        cost = torch.zeros_like(X)
    else:
        # Reshape for potential function evaluation
        # flat_xy = full_dim_xy.reshape(-1, D)
        
        # Evaluate first potential
        flat_cost = spline.potential[0](full_dim_xy)
        
        # Add other potentials
        for i in range(1, len(spline.potential)):
            flat_cost += spline.potential[i](full_dim_xy)
        
        # Reshape back to grid form - ensure flat_cost is the right size
        cost = flat_cost.reshape(X.shape[0], X.shape[1])#.detach()

    # Use autumn for the potential as specified
    potential_cmap = plt.cm.autumn
    contour = plt.contourf(X, Y, cost.cpu().detach(), levels=100, alpha=0.6)
    cbar_pot = plt.colorbar(contour, label='Potential Energy')

    # Now plot the samples. 

    for i in range(T):

        plt.scatter(
            xt[:, i, 0],
            xt[:,i,1],
            c=[colors[i]], s = 10,alpha = 0.7, edgecolor='none'
        )

    # Add a colorbar for the trajectories

    sm = plt.cm.ScalarMappable(cmap = custom_cmap, norm = plt.Normalize(0,1))
    sm.set_array([])

    cbar_traj = plt.colorbar(sm,ax = ax,label = 'Time (t)')
    cbar_traj.set_ticks(np.linspace(0, 1, 5))
    cbar_traj.set_ticklabels([f'{t:.2f}' for t in np.linspace(0, 1, 5)])

    # Add grid and labels
    plt.grid(alpha=0.2, linestyle='--', color='white')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Optimal Transport Path Through Potential Field', fontsize=14)
    
    
    plt.tight_layout()


def path_visualization(interpolation, arch, spline, x0, y0, x1, y1, 
                                     num_samples=1000, time_steps=10, solver='euler', 
                                     z=None, num_contour_points=200,idx_x = 0,idx_y = 1):
    """
    Visualize the path of the interpolation in the sample space.
    interpolation: torch.tensor, shape (1,s,arch)
    arch: list, architecture of the model [dim,width,num_layers]
    num_samples: int, number of samples to visualize
    time_steps: int, number of time steps for integrating NODEs
    solver: str, solver for integrating NODEs: 'euler' for energy, 'dopri5' for accuracy
    -----
    return: torch.tensor, shape (s,num_samples,dim_samp)
    """
    from matplotlib.colors import LinearSegmentedColormap
    device = interpolation.device
    s = interpolation.shape[1]
    dim_th = interpolation.shape[-1]
    dim_samp = arch[0]
    
    # Set up samples
    if z is None:
        if num_samples is None:
            num_samples = 1000
        # Use prior distribution to generate samples if not provided
        try:
            z = torch.randn(num_samples, arch[0]).to(device)
        except:
            raise ValueError("Cannot create samples, provide either num_samples or z")
    else:
        num_samples = z.shape[0]
    
    # Determine data dimensionality
    D = z.shape[1]
    
    
    # Output container for samples
    samples_path = torch.zeros(num_samples, s, D, device=device)
    
    # For each time step, compute the sample trajectory
    for i in range(s):
        theta = interpolation[:, i, :].squeeze()
        samples = spline.push_forward(theta,z,t_node = time_steps)
        samples_path[:, i, :] = samples.detach().cpu() #[-1, :, :]
    
    # Create a custom colormap for time steps
    colors = plt.cm.autumn(np.linspace(0, 1, s))
    custom_cmap = LinearSegmentedColormap.from_list('autumn', colors)
    
    # Create visualization with potential function
    fig,ax = plt.subplots(figsize=(12, 10))
    
    # First plot the potential function
    X, Y = torch.meshgrid(
        torch.linspace(x0, x1, num_contour_points),
        torch.linspace(y0, y1, num_contour_points),
        indexing='ij'  # Ensure consistent indexing
    )
    num_points = X.shape[0] * X.shape[1]
    # Create a full-dimensional input tensor but only populate first two dimensions
    full_dim_xy = torch.zeros(num_points,1,D, device=device)
    full_dim_xy[:,0, 0] = X.reshape(num_points).flatten()
    full_dim_xy[:,0, 1] = Y.reshape(num_points).flatten()

    

    if spline.potential is None:
        cost = torch.zeros_like(X)
    else:
        # Reshape for potential function evaluation
        # flat_xy = full_dim_xy.reshape(-1, D)
        
        # Evaluate first potential
        flat_cost = spline.potential[0](full_dim_xy)
        
        # Add other potentials
        for i in range(1, len(spline.potential)):
            flat_cost += spline.potential[i](full_dim_xy)
        
        # Reshape back to grid form - ensure flat_cost is the right size
        cost = flat_cost.reshape(X.shape[0], X.shape[1])#.detach()

    # Use autumn for the potential as specified
    potential_cmap = plt.cm.autumn
    contour = plt.contourf(X, Y, cost.cpu().detach(), levels=100, alpha=0.6) #, cmap=potential_cmap
    cbar_pot = plt.colorbar(contour, label='Potential Energy')
    # Then overlay the trajectory
    # Plot the first two dimensions regardless of original dimensionality
    
    for i in range(s):
        if i == 0:
            plt.scatter(
                samples_path[:, i, idx_x].cpu().numpy(), 
                samples_path[:, i, idx_y].cpu().numpy(),
                c=[colors[i]], s=20, alpha=0.9, label=f'ρ₀ (t=0)', edgecolor='white', linewidth=0.5)
        elif i == s-1:
            plt.scatter(
                samples_path[:, i, idx_x].cpu().numpy(), 
                samples_path[:, i, idx_y].cpu().numpy(),
                c=[colors[i]], s=20, alpha=0.9, label=f'ρ₁ (t=1)', edgecolor='white', linewidth=0.5)
        else:
            plt.scatter(
                samples_path[:, i, idx_x].cpu().numpy(), 
                samples_path[:, i, idx_y].cpu().numpy(),
                c=[colors[i]], s=10, alpha=0.7, edgecolor='none')
    
    # Add a colorbar for the trajectories
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    
    cbar_traj = plt.colorbar(sm, ax= ax, label='Time (t)')
    cbar_traj.set_ticks(np.linspace(0, 1, 5))
    cbar_traj.set_ticklabels([f'{t:.2f}' for t in np.linspace(0, 1, 5)])
        
    # Add grid and labels
    plt.grid(alpha=0.2, linestyle='--', color='white')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Optimal Transport Path Through Potential Field', fontsize=14)
    
    
    plt.tight_layout()
    
    return samples_path

def path_visualization_snapshots(interpolation, arch, spline, x0, y0, x1, y1, 
                                num_samples=1000, time_steps=10, solver='euler', 
                                z=None, num_contour_points=200, idx_x=0, idx_y=1):
    """
    Visualize the path of the interpolation in the sample space with 6 snapshots.
    
    Parameters:
    -----------
    interpolation: torch.tensor, shape (1,s,arch)
        The interpolation points at different time steps
    arch: list
        Architecture of the model [dim,width,num_layers]
    spline: Spline object
        The spline model used for pushforward
    x0, y0, x1, y1: float
        Bounds for plotting
    num_samples: int
        Number of samples to visualize
    time_steps: int
        Number of time steps for integrating NODEs
    solver: str
        Solver for integrating NODEs: 'euler' for energy, 'dopri5' for accuracy
    z: torch.tensor, optional
        Pre-generated samples to use instead of random generation
    num_contour_points: int
        Number of points to use for potential contour plotting
    idx_x, idx_y: int
        Indices of dimensions to plot
        
    Returns:
    --------
    torch.tensor, shape (num_samples, s, dim_samp)
        The trajectories of the samples
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    
    device = interpolation.device
    s = interpolation.shape[1]
    dim_th = interpolation.shape[-1]
    dim_samp = arch[0]
    
    # Set up samples
    if z is None:
        if num_samples is None:
            num_samples = 1000
        # Use prior distribution to generate samples if not provided
        try:
            z = torch.randn(num_samples, arch[0]).to(device)
        except:
            raise ValueError("Cannot create samples, provide either num_samples or z")
    else:
        num_samples = z.shape[0]
    
    # Determine data dimensionality
    D = z.shape[1]
    
    # Output container for samples
    samples_path = torch.zeros(num_samples, s, D, device=device)
    
    # For each time step, compute the sample trajectory
    for i in range(s):
        theta = interpolation[:, i, :].squeeze()
        samples = spline.push_forward(theta, z, t_node=time_steps)
        samples_path[:, i, :] = samples.detach().cpu()
    
    # Create a custom colormap for time steps
    colors = plt.cm.autumn(np.linspace(0, 1, s))
    custom_cmap = LinearSegmentedColormap.from_list('autumn', colors)
    
    # Compute the potential field once for efficiency
    X, Y = torch.meshgrid(
        torch.linspace(x0, x1, num_contour_points),
        torch.linspace(y0, y1, num_contour_points),
        indexing='ij'
    )
    num_points = X.shape[0] * X.shape[1]
    full_dim_xy = torch.zeros(num_points, 1, D, device=device)
    full_dim_xy[:, 0, idx_x] = X.reshape(num_points).flatten()
    full_dim_xy[:, 0, idx_y] = Y.reshape(num_points).flatten()

    if spline.potential is None:
        cost = torch.zeros_like(X)
    else:
        flat_cost = spline.potential[0](full_dim_xy)
        
        for i in range(1, len(spline.potential)):
            flat_cost += spline.potential[i](full_dim_xy)
        
        cost = flat_cost.reshape(X.shape[0], X.shape[1])
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # Divide the trajectory into 6 segments
    time_intervals = np.linspace(0, s-1, 7).astype(int)
    
    for snapshot in range(6):
        start_idx = time_intervals[snapshot]
        end_idx = time_intervals[snapshot+1]
        
        # Calculate the real time interval (0-1)
        start_time = start_idx / (s-1)
        end_time = end_idx / (s-1)
        
        ax = fig.add_subplot(gs[snapshot // 3, snapshot % 3])
        
        # Plot the potential field
        contour = ax.contourf(X, Y, cost.cpu().detach(), levels=100, alpha=0.6) #, cmap=plt.cm.autumn
        
        # Plot samples for this time segment with colors reflecting progression
        segment_colors = colors[start_idx:end_idx+1]
        segment_times = np.linspace(start_time, end_time, end_idx - start_idx + 1)
        
        for i, t_idx in enumerate(range(start_idx, end_idx + 1)):
            if t_idx == 0:  # Initial distribution
                ax.scatter(
                    samples_path[:, t_idx, idx_x].cpu().numpy(),
                    samples_path[:, t_idx, idx_y].cpu().numpy(),
                    c=[segment_colors[i]], s=20, alpha=0.9, 
                    label=f'ρ₀ (t=0)', edgecolor='white', linewidth=0.5
                )
            elif t_idx == s-1:  # Final distribution
                ax.scatter(
                    samples_path[:, t_idx, idx_x].cpu().numpy(),
                    samples_path[:, t_idx, idx_y].cpu().numpy(),
                    c=[segment_colors[i]], s=20, alpha=0.9, 
                    label=f'ρ₁ (t=1)', edgecolor='white', linewidth=0.5
                )
            else:
                ax.scatter(
                    samples_path[:, t_idx, idx_x].cpu().numpy(),
                    samples_path[:, t_idx, idx_y].cpu().numpy(),
                    c=[segment_colors[i]], s=10, alpha=0.7, edgecolor='none'
                )
        
        # Add grid and labels
        ax.grid(alpha=0.2, linestyle='--', color='white')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'Transport Path: t ∈ [{start_time:.2f}, {end_time:.2f}]', fontsize=14)
        
        # Add legend only for the first and last snapshots
        if snapshot == 0 or snapshot == 5:
            ax.legend(loc='upper right', fontsize=10)
    
    # Add a colorbar for the potential
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, label='Potential Energy')
    
    # Add a colorbar for time progression
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    time_cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    time_cbar = fig.colorbar(sm, cax=time_cbar_ax, orientation='horizontal', label='Time (t)')
    time_cbar.set_ticks(np.linspace(0, 1, 7))
    time_cbar.set_ticklabels([f'{t:.2f}' for t in np.linspace(0, 1, 7)])
    
    plt.suptitle('Optimal Transport Path Through Potential Field - Time Evolution', fontsize=16)
    plt.tight_layout(rect=[0, 0.07, 0.9, 0.95])
    
    return samples_path




def path_visualization_with_trajectories(interpolation, arch, spline, x0, y0, x1, y1, 
                                        num_samples=1000, time_steps=10, solver='euler', 
                                        z=None, num_contour_points=200, idx_x=0, idx_y=1, 
                                        show_trajectories=True):
    """
    Visualize the path of the interpolation with optional sample trajectories.
    
    Parameters:
    -----------
    (Same parameters as path_visualization_snapshots)
    show_trajectories: bool
        Whether to plot the trajectories of individual particles
        
    Returns:
    --------
    torch.tensor, shape (num_samples, s, dim_samp)
        The trajectories of the samples
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    device = interpolation.device
    s = interpolation.shape[1]
    dim_th = interpolation.shape[-1]
    dim_samp = arch[0]
    
    # Set up samples
    if z is None:
        if num_samples is None:
            num_samples = 1000
        try:
            z = torch.randn(num_samples, arch[0]).to(device)
        except:
            raise ValueError("Cannot create samples, provide either num_samples or z")
    else:
        num_samples = z.shape[0]
    
    # Determine data dimensionality
    D = z.shape[1]
    
    # Output container for samples
    samples_path = torch.zeros(num_samples, s, D, device=device)
    
    # For each time step, compute the sample trajectory
    for i in range(s):
        theta = interpolation[:, i, :].squeeze()
        samples = spline.push_forward(theta, z, t_node=time_steps)
        samples_path[:, i, :] = samples.detach().cpu()
    
    # Create a custom colormap for time steps
    colors = plt.cm.autumn(np.linspace(0, 1, s))
    custom_cmap = LinearSegmentedColormap.from_list('autumn', colors)
    
    # Create visualization with potential function
    fig,ax = plt.subplots(figsize=(12, 10))
    
    # Compute potential field
    X, Y = torch.meshgrid(
        torch.linspace(x0, x1, num_contour_points),
        torch.linspace(y0, y1, num_contour_points),
        indexing='ij'
    )
    num_points = X.shape[0] * X.shape[1]
    full_dim_xy = torch.zeros(num_points, 1, D, device=device)
    full_dim_xy[:, 0, idx_x] = X.reshape(num_points).flatten()
    full_dim_xy[:, 0, idx_y] = Y.reshape(num_points).flatten()

    if spline.potential is None:
        cost = torch.zeros_like(X)
    else:
        flat_cost = spline.potential[0](full_dim_xy)
        
        for i in range(1, len(spline.potential)):
            flat_cost += spline.potential[i](full_dim_xy)
        
        cost = flat_cost.reshape(X.shape[0], X.shape[1])
    
    # Plot potential field
    contour = plt.contourf(X, Y, cost.cpu().detach(), levels=100, alpha=0.6) #
    cbar_pot = plt.colorbar(contour, label='Potential Energy')
    
    # If showing trajectories, plot lines connecting samples across time
    if show_trajectories:
        # Select a smaller subset of particles to show trajectories for
        trajectory_indices = np.random.choice(num_samples, size=min(100, num_samples), replace=False)
        for idx in trajectory_indices:
            plt.plot(
                samples_path[idx, :, idx_x].cpu().numpy(),
                samples_path[idx, :, idx_y].cpu().numpy(),
                '-', linewidth=0.25, alpha=0.3, color='gray'
            )
    
    # Plot the sample distributions at each time step
    for i in range(s):
        if i == 0:
            plt.scatter(
                samples_path[:, i, idx_x].cpu().numpy(),
                samples_path[:, i, idx_y].cpu().numpy(),
                c=[colors[i]], s=20, alpha=0.9, label=f'ρ₀ (t=0)', edgecolor='white', linewidth=0.5
            )
        elif i == s-1:
            plt.scatter(
                samples_path[:, i, idx_x].cpu().numpy(),
                samples_path[:, i, idx_y].cpu().numpy(),
                c=[colors[i]], s=20, alpha=0.9, label=f'ρ₁ (t=1)', edgecolor='white', linewidth=0.5
            )
        else:
            plt.scatter(
                samples_path[:, i, idx_x].cpu().numpy(),
                samples_path[:, i, idx_y].cpu().numpy(),
                c=[colors[i]], s=10, alpha=0.7, edgecolor='none'
            )
    
    # Add a colorbar for the trajectories
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_traj = plt.colorbar(sm,ax = ax, label='Time (t)')
    cbar_traj.set_ticks(np.linspace(0, 1, 5))
    cbar_traj.set_ticklabels([f'{t:.2f}' for t in np.linspace(0, 1, 5)])
    
    # Add grid and labels
    plt.grid(alpha=0.2, linestyle='--', color='white')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Optimal Transport Path Through Potential Field with Trajectories', fontsize=14)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    return samples_path


class TrajectoryVisualizer:
    """
    A class for visualizing high-dimensional trajectory data using various 
    dimensionality reduction techniques.
    CODE FROM claude3.5
    """
    
    def __init__(self, trajectory_data):
        """
        Initialize with trajectory data of shape (n_timesteps, n_dimensions)
        """
        self.data = trajectory_data
        self.n_timesteps, self.n_dims = trajectory_data.shape
        
    def reduce_dimensions(self, method='pca', n_components=2, **kwargs):
        """
        Reduce dimensionality using specified method
        """
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        
        elif method.lower() == 'mds':
            reducer = MDS(n_components=n_components, **kwargs)
        elif method.lower() == 'isomap':
            reducer = Isomap(n_components=n_components, **kwargs)
        else:
            raise ValueError("Unsupported reduction method")
            
        return reducer.fit_transform(self.data)
    
    def plot_trajectory(self, reduced_data, method_name, ax=None, show_points=True,
                       cmap='viridis', arrow_freq=5):
        """
        Plot reduced trajectory with direction arrows
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            if reduced_data.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        # Plot the trajectory line
        if reduced_data.shape[1] == 3:
            ax.plot(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                   alpha=0.5, label='Trajectory')
            if show_points:
                scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                   c=range(len(reduced_data)), cmap=cmap)
        else:
            ax.plot(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5,
                   label='Trajectory')
            if show_points:
                scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                                   c=range(len(reduced_data)), cmap=cmap)
        
        # Add direction arrows
        for i in range(0, len(reduced_data)-1, arrow_freq):
            if reduced_data.shape[1] == 3:
                ax.quiver(reduced_data[i, 0], reduced_data[i, 1], reduced_data[i, 2],
                         reduced_data[i+1, 0] - reduced_data[i, 0],
                         reduced_data[i+1, 1] - reduced_data[i, 1],
                         reduced_data[i+1, 2] - reduced_data[i, 2],
                         color='red', alpha=0.5)
            else:
                ax.quiver(reduced_data[i, 0], reduced_data[i, 1],
                         reduced_data[i+1, 0] - reduced_data[i, 0],
                         reduced_data[i+1, 1] - reduced_data[i, 1],
                         color='red', alpha=0.5)
        
        ax.set_title(f'Trajectory Visualization using {method_name}')
        if show_points:
            plt.colorbar(scatter, label='Time Step')
        
        return ax
    
    def plot_multiple_views(self, methods=['pca', 'tsne', 'umap'],
                          n_components=2, **kwargs):
        """
        Create multiple visualizations using different reduction methods
        """
        n_methods = len(methods)
        fig = plt.figure(figsize=(6*n_methods, 5))
        
        for i, method in enumerate(methods):
            reduced_data = self.reduce_dimensions(method, n_components, **kwargs)
            ax = fig.add_subplot(1, n_methods, i+1)
            self.plot_trajectory(reduced_data, method.upper(), ax=ax)
        
        plt.tight_layout()
        return fig
    

def plot_hist(lagrangian_history, potential_history,bd0_distance,bd1_distance, figures_dir):
    """Plot the history of the Lagrangian during optimization."""
    sns.set_theme(style="darkgrid")

    # First, let's inspect and correctly prepare the data
    # Print the shape to debug
    print(f"Lagrangian history type: {type(lagrangian_history)}")
    if len(lagrangian_history) > 0:
        print(f"First element type: {type(lagrangian_history[0])}, shape: {np.array(lagrangian_history[0]).shape if hasattr(lagrangian_history[0], 'shape') else 'no shape'}")

    # Properly flatten data regardless of structure
    flat_data = []
    for item in lagrangian_history:
        if isinstance(item, (list, np.ndarray)):
            # If item is a list or array, extend with all its values
            if isinstance(item, np.ndarray) and item.ndim > 1:
                # Handle multi-dimensional arrays
                flat_data.extend(item.flatten())
            else:
                flat_data.extend(item)
        else:
            # If item is a scalar, append it
            flat_data.append(item)

    # Convert to numpy array for plotting
    lagrangian_data = np.array(flat_data)

    # Now plot with seaborn
    plt.figure(figsize=(12, 6))
    # Use seaborn lineplot without DataFrame
    sns.lineplot(x=range(len(lagrangian_data)), y=lagrangian_data, linewidth=2)

    plt.title('Lagrangian History During Optimization', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Lagrangian Value', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "lagrangian_history.png"), dpi=300)
    plt.close()

    # Properly flatten potential_history data regardless of structure
    flat_potential_data = []
    for item in potential_history:
        if isinstance(item, (list, np.ndarray)):
            # If item is a list or array, extend with all its values
            if isinstance(item, np.ndarray) and item.ndim > 1:
                # Handle multi-dimensional arrays
                flat_potential_data.extend(item.flatten())
            else:
                flat_potential_data.extend(item)
        else:
            # If item is a scalar, append it
            flat_potential_data.append(item)

    # Convert to numpy array for plotting
    potential_data = np.array(flat_potential_data)

    # Now plot with seaborn
    plt.figure(figsize=(12, 6))
    # Use seaborn lineplot without DataFrame
    sns.lineplot(x=range(len(potential_data)), y=potential_data, linewidth=2)

    plt.title('Potential Energy History During Optimization', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Potential Energy', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "potential_history.png"), dpi=300)
    plt.close()

    # Plot accuracy of representation of boundary conditions
    opt_steps = len(lagrangian_data)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=np.linspace(0, opt_steps, len(bd0_distance)), y=np.array(bd0_distance), label='Source Distribution', linewidth=2)
    sns.lineplot(x=np.linspace(0, opt_steps, len(bd1_distance)), y=np.array(bd1_distance), label='Target Distribution', linewidth=2)
    plt.title('Boundary Condition Accuracy During Optimization', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "boundary_accuracy.png"), dpi=300)
    plt.close()
    


def plot_pot(spline,x0,y0,x1,y1,num_points=1000):
    '''
    Input:
        spline: torch.nn.Module
        x0: float
        y0: float
        x1: float
        y1: float
        num_points: int
    Output:
        None
    '''

    X,Y = torch.meshgrid(torch.linspace(x0,x1,num_points),torch.linspace(y0,y1,num_points))
    xy = torch.stack([X,Y],dim = -1)
    if spline.potential is None:
        cost = torch.zeros_like(X)
    else:
        # cost = spline.potential(xy)
        cost = spline.potential[0](xy)
        for i in range(1,len(spline.potential)):
            cost += spline.potential[i](xy)
    
    plt.contourf(X,Y,cost,levels = 100,alpha = 0.5)

    # plt.show()

    return 


def create_particle_animation(spline,samples_path,x0,x1,y0,y1, interval=50):
        """
        Creates an animation of particles evolving over time.
        
        Args:
            samples_path: tensor of shape (timesteps, num_particles, 2) containing particle positions
            interval: time in milliseconds between frames
        
        Returns:
            HTML animation object
        """
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_pot(spline,x0,y0,x1,y1,100)
        scatter = ax.scatter(samples_path[0, :, 0], samples_path[0, :, 1], s=1)
        
        ax.set_xlim([samples_path[:, :, 0].min(), samples_path[:, :, 0].max()])#samples_path[:, :, 0].min(), samples_path[:, :, 0].max()
        ax.set_ylim([samples_path[:, :, 1].min(), samples_path[:, :, 1].max()])#samples_path[:, :, 1].min(), samples_path[:, :, 1].max()
        
        # Create a colormap
        cmap = plt.get_cmap('autumn')
        norm = plt.Normalize(vmin=0, vmax=len(samples_path))

        def update(frame):
            colors = cmap(norm(frame))
            scatter.set_offsets(samples_path[frame, :, :2])
            scatter.set_color(colors)
            return scatter,
        
        anim = FuncAnimation(fig, update, frames=len(samples_path), 
                            interval=interval, blit=True)
        plt.close()
        
        return HTML(anim.to_jshtml())

