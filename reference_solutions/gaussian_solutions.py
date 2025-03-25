import torch as torch
import torch.nn.functional as F
import numpy as np
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt

def sqrt_cov(sig: torch.tensor, inverse= False):
  '''
  sig: nxn SPD matrix
  '''
  D,U = torch.linalg.eig(sig)
  if inverse:
    return torch.matmul(torch.matmul(U,torch.diag(D**0.5)),torch.linalg.inv(U)).type(torch.float64),torch.matmul(torch.matmul(U,torch.diag(D**(-0.5))),torch.linalg.inv(U)).type(torch.float64)
  else:
    return torch.matmul(torch.matmul(U,torch.diag(D**0.5)),torch.linalg.inv(U)).type(torch.float64)


def W2_gaussian(sig1,sig2,mu1 = None,mu2 = None):
    '''
    mu1, mu2: means
    sig1, sig2: covariances
    '''
    if mu1 is None:
        mu1 = torch.zeros(sig1.shape[0])
    if mu2 is None:
        mu2 = torch.zeros(sig2.shape[0])
    sig1_sqrt = sqrt_cov(sig1)
    sig1_sqrt = sig1_sqrt.type(mu1.dtype)
    temp = torch.matmul(torch.matmul(sig1_sqrt,sig2),sig1_sqrt)
    return (torch.norm(mu1-mu2)**2 + torch.trace(sig1+sig2-2*sqrt_cov(temp)))**(1/2)


def mean_covmat(x):
    '''
    x: bxn
    '''
    mean = torch.mean(x,dim=0)
    x_centered = x - mean
    x_centered = x_centered.unsqueeze(2)
    x_centered_T = x_centered.permute(0,2,1)
    cov = torch.matmul(x_centered,x_centered_T)

    return mean,cov.mean(dim=0)
    

def linear_interp(sig0,sig1,t):
    '''
    sig0, sig1: nxn SPD matrices
    t: linspace(0,1)
    '''
    tt = t.unsqueeze(1).unsqueeze(2)  # b x 1 x 1
    
    gamma0,gamma0_inv = sqrt_cov(sig0,inverse=True)

    gamma0 = gamma0.type(torch.float32)
    gamma0_inv = gamma0_inv.type(torch.float32)

    mat = sqrt_cov(gamma0@sig1@gamma0,inverse = False).type(torch.float32)

    gamma_t = gamma0*(1-tt)+tt*gamma0_inv@mat

    gamma_t_T = gamma_t.permute(0,2,1)

    return torch.matmul(gamma_t,gamma_t_T)

def monge_map(sig0,sig1,t,z,mu0 = None,mu1 = None):
    '''
    sig0, sig1: nxn SPD matrices
    t: linspace(0,1)
    z: bsxn vector samples rho0
    mu0, mu1: means (nx1)
    '''
    if mu0 is None:
        mu0 = torch.zeros(sig0.shape[0],1).to(sig0)
    if mu1 is None:
        mu1 = torch.zeros(sig1.shape[0],1).to(sig1)

    gamma0,gamma0_inv = sqrt_cov(sig0,inverse=True)
    gamma0 = gamma0.type(torch.float32)
    gamma0_inv = gamma0_inv.type(torch.float32)
    mat = sqrt_cov(gamma0@sig1@gamma0,inverse = False).type(torch.float32)
    monge = gamma0_inv@mat@gamma0_inv

    monge_z = torch.matmul(z-mu0,monge.T)+mu1

    tt = t.unsqueeze(1).unsqueeze(2)  # b x 1 x 1

    return monge_z*(tt)+(1-tt)*z


def sample_cov_mats(k,n):
  sampling_cov_lkj = torch.distributions.lkj_cholesky.LKJCholesky(n,1)
  cov_mats_lkj = []
  for i in range(k):
    L = sampling_cov_lkj.sample()#.to(torch.float64)
    cov_mats_lkj.append(L@L.T)
  return cov_mats_lkj

# L2-UVP
def monge_map_torch(sig0,sig1,x0):

    gamma0,gamma0_inv = sqrt_cov(sig0,inverse=True)
    gamma0 = gamma0.type(torch.float32)
    gamma0_inv = gamma0_inv.type(torch.float32)
    mat = sqrt_cov(gamma0@sig1@gamma0,inverse = False).type(torch.float32)
    monge = gamma0_inv@mat@gamma0_inv

    monge_z = torch.matmul(x0,monge.T)

    return monge_z

def l2_uvp(model0,model1,rho0,rho1,sig0,sig1,device,num_samples = 10_000):

    x0 = rho0.sample((num_samples,)).to(device)

    x1 = monge_map_torch(sig0,sig1,x0)

    var_x1 = torch.sum(torch.var(x1,dim = 0))

    z = model0(x0)
    x1_hat = model1(z,reverse = True)


    l2_uvp = torch.mean((x1_hat-x1).norm(dim = 1)**2)/(var_x1)

    return l2_uvp*100


def sqrt_matrix(matrix):
    """
    Compute the matrix square root using numpy's eigen decomposition and converting back to torch.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Input symmetric positive definite matrix
        
    Returns:
    --------
    torch.Tensor
        Square root of the input matrix
        
    """
    # Store original device and dtype
    original_device = matrix.device
    original_dtype = matrix.dtype
    
    # Convert to numpy array
    matrix_np = matrix.cpu().detach().numpy()
    
    # Compute eigendecomposition using numpy
    # numpy.linalg.eigh is more stable for symmetric matrices
    eig_values_np, eig_vectors_np = np.linalg.eigh(matrix_np)
    
    # Take square root of eigenvalues
    sqrt_eig_values_np = np.sqrt(eig_values_np)
    
    # Compute matrix square root: Q * sqrt(Λ) * Q^T
    sqrt_matrix_np = eig_vectors_np @ np.diag(sqrt_eig_values_np) @ eig_vectors_np.T
    
    # Convert back to torch tensor with original properties
    sqrt_matrix_torch = torch.from_numpy(sqrt_matrix_np).to(
        dtype=original_dtype,
        device=original_device
    )
    
    return sqrt_matrix_torch

def is_symmetric_positive_definite(matrix, tolerance=1e-8):
    """
    Check if input matrix is symmetric positive definite
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Input matrix to check
    tolerance : float
        Numerical tolerance for eigenvalue positivity check
        
    Returns:
    --------
    bool
        True if matrix is symmetric positive definite
    """
    # Check symmetry
    if not torch.allclose(matrix, matrix.t()):
        return False
    
    # Convert to numpy for eigenvalue check
    matrix_np = matrix.cpu().detach().numpy()
    
    # Compute eigenvalues
    try:
        eig_values = np.linalg.eigvalsh(matrix_np)
        return np.all(eig_values > tolerance)
    except np.linalg.LinAlgError:
        return False



def exact_solution_Gaussian_SB(sig0,sig1,mu0,mu1,t,sigma):
    '''
    Input:
        sig0: [d,d] tensor
        sig1: [d,d] tensor
        mu0: [d] tensor
        mu1: [d] tensor
        t: [N] tensor
        sigma: [1] tensor
    Output:
        sigt : [N,d,d] tensor
    '''
    d = sig0.shape[0]
    
    sqrt_sig0 = sqrt_matrix(sig0)
    inv_sqrt_sig0 = torch.linalg.inv(sqrt_sig0)
    D_sig = sqrt_matrix(4*sqrt_sig0 @ sig1 @ sqrt_sig0+sigma**4*torch.eye(d).to(sig0.device))
    C_sig = (sqrt_sig0@D_sig@inv_sqrt_sig0-sigma**2*torch.eye(d).to(sig0.device))/2
    tt = t.view(-1,1).unsqueeze(-1)
    sigt = (1-tt)**2*sig0+tt**2*sig1+tt*(1-tt)*(C_sig+C_sig.T+sigma**2*torch.eye(d).to(sig0.device))

    mut = (1-tt)*mu0+tt*mu1

    return mut,sigt

def fisher_info_gaussian(mean_path,cov_path,time_disc,sigma = 1):
    '''
    Input:
    mean_path: [N,d] tensor
    cov_path: [N,d,d] tensor
    time_disc: [N] tensor
    Output:
    fisher_information: [1] tensor
    
    The Fisher information of a path is defined by
    \int_0^1 E[||\nabla_x \rho(t,x)||^2;\rho(t,x)] dt
    where \rho(t,x) is the density of the path at time t and x
    For a Gaussian densities, the Fisher information is given by
    \int_0^1 E[||\Sigma_t^{-1}(x-mu_t)||^2;\rho(t,x)] dt = \int_0^1 tr(\Sigma_t^{-1}) dt
    where mu_t and Sigma_t are the mean and covariance of the density at time t
    '''
    
    fisher_path = torch.tensor([torch.trace(torch.linalg.inv(cov_path[i])) for i in range(cov_path.shape[0])]).to(mean_path.device)
    
    fisher_information = torch.trapz(fisher_path,time_disc)*sigma**4/8
    return fisher_information,fisher_path


def estimate_mean_cov(path):
    '''
    Input
    path: [Bs,N,d] tensor
    Output
    mean: [N,d] tensor
    cov: [N,d,d] tensor
    '''
    Bs,N,d = path.shape
    mean = path.mean(0)
    cov = torch.zeros(N,d,d).to(path.device)
    for i in range(N):
        cov[i] = torch.cov(path[:,i,:].T)
    return mean,cov

def plot_confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of a 2D Gaussian distribution
    
    Parameters
    ----------
    mean : array_like, shape (2, )
        Mean of the distribution
    cov : array_like, shape (2, 2)
        Covariance matrix of the distribution
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    
    # Get eigenvalues and eigenvectors of the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Make sure eigenvalues are positive (sometimes numerical issues)
    eigenvals = np.maximum(eigenvals, 1e-10)
    
    # Get the index of the largest eigenvalue
    largest_eigval_idx = np.argmax(eigenvals)
    largest_eigval = eigenvals[largest_eigval_idx]
    smallest_eigval = eigenvals[1 - largest_eigval_idx]
    
    # Calculate the angle using the eigenvector of the largest eigenvalue
    angle = np.arctan2(eigenvecs[1, largest_eigval_idx], eigenvecs[0, largest_eigval_idx])
    angle_deg = np.degrees(angle)
    
    # Width and height are related to the eigenvalues and n_std
    width = 2 * n_std * np.sqrt(largest_eigval)
    height = 2 * n_std * np.sqrt(smallest_eigval)
    # Create and add the ellipse
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle_deg,
        fill=False,
        **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse
    


def test_results_gaussian_SB(mu0, mu1, sig0, sig1, sig, t,spline, sample_path, l2_errors=True, plot=True, figsize=(12, 10)):
    """
    Evaluate and visualize the accuracy of a sample path compared to the exact solution
    for a Gaussian Schrödinger Bridge problem.
    
    Parameters:
    -----------
    mu0, mu1: tensors of shape [d]
        Mean vectors for the initial and final distributions
    sig0, sig1: tensors of shape [d,d]
        Covariance matrices for the initial and final distributions
    sig: tensor of shape [1]
        Diffusion parameter for the Schrödinger Bridge
    t: tensor of shape [N]
        Time points at which to evaluate
    spline: spline class
        Function that takes a tensor of shape [N] and returns a tensor of shape [N,d]
    sample_path: tensor of shape [Bs,N,d]
        Sample trajectories from the learned model
    l2_errors: bool
        Whether to compute L2 errors 
    plot: bool
        Whether to create visualization plots
    figsize: tuple
        Figure size for the plots
    
    Returns:
    --------
    Various error metrics comparing the sample path to the exact solution
    """
    device = sample_path.device
    sample_path = sample_path.permute(1, 0, 2)
    N, Bs, d = sample_path.shape

    # Compute the mean and covariance of the sample path
    sol_mean, sol_cov = estimate_mean_cov(sample_path.permute(1, 0, 2))
    # make sol_mean same shape as the next shapes (N,1,d)
    sol_mean = sol_mean.unsqueeze(1)
    
    # Exact solution
    mu_t, cov_t = exact_solution_Gaussian_SB(sig0, sig1, mu0, mu1, t, sig)
    
    # Exact approx path
    mu_t2, cov_t2 = exact_solution_Gaussian_SB(sol_cov[0], sol_cov[-1], sol_mean[0], sol_mean[-1], t, sig)

    sol_mean = sol_mean.detach()
    sol_cov = sol_cov.detach()
    mu_t = mu_t.detach()
    cov_t = cov_t.detach()
    mu_t2 = mu_t2.detach()
    cov_t2 = cov_t2.detach()

    # Compute the Wasserstein distance
    w2_error_true_path = torch.zeros(t.shape[0])
    w2_error_approx_path = torch.zeros(t.shape[0])
    
    if l2_errors:
        l2_mean_true = torch.norm(mu_t.view(-1, d) - sol_mean.view(-1, d), dim=1) / (torch.norm(mu_t.view(-1, d), dim=1) + 1e-8)
        l2_cov_true = torch.linalg.matrix_norm(cov_t.view(-1, d, d) - sol_cov) / (torch.linalg.matrix_norm(cov_t) + 1e-8)
        l2_mean_approx = torch.norm(mu_t2.view(-1, d) - sol_mean.view(-1, d), dim=1) / (torch.norm(mu_t2.view(-1, d), dim=1) + 1e-8)
        l2_cov_approx = torch.linalg.matrix_norm(cov_t2.view(-1, d, d) - sol_cov) / (torch.linalg.matrix_norm(cov_t2) + 1e-8)

    for i in range(t.shape[0]):
        w2_error_true_path[i] = W2_gaussian(cov_t[i], sol_cov[i], mu_t[i], sol_mean[i])
        w2_error_approx_path[i] = W2_gaussian(cov_t2[i], sol_cov[i], mu_t2[i], sol_mean[i])

    if plot and d >= 2:  # Only plot if it's at least 2D
        # Choose representative time points to plot
        if N <= 5:
            time_indices = list(range(N))
        else:
            # Always include the first and last points, plus evenly spaced interior points
            split = max(1, N // 4)
            time_indices = [0] + [i for i in range(split, N-split, max(1, (N-2*split)//3))] + [N-1]
        
        
        num_plots = len(time_indices)
        nrows = (num_plots + 2) // 3  # Calculate rows needed for time points + 3 error plots
        ncols = min(3, num_plots)
        
        part_batch = 5
        z = torch.randn(Bs // part_batch, d).to(device)
        
        # Create figure for distribution plots
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

        # Make sure axs is always a 2D array, even if nrows or ncols is 1
        axs = np.atleast_2d(axs)

        # Create a flattened list of valid (idx, i, ax) tuples for plotting
        plot_items = []
        for idx, i in enumerate(time_indices):
            # Calculate the position in the grid
            row_idx = idx // ncols
            col_idx = idx % ncols
            
            # Only add if within bounds of the grid
            if row_idx < nrows and col_idx < ncols:
                plot_items.append((idx, i, axs[row_idx, col_idx]))

        # Get parameters for plotting times
        theta_t = spline(t[time_indices]).squeeze(0)

        # Plot distributions at selected time points
        for idx, i, ax in plot_items:
            
            # Generate samples
            sample_path_t = mu_t[i].unsqueeze(-1) + torch.linalg.cholesky(cov_t[i]) @ z.T
            sample_path_t2 = mu_t2[i].unsqueeze(-1) + torch.linalg.cholesky(cov_t2[i]) @ z.T
            # Obtain z from ref density
            if i == 0:
                
                z0 = spline.pull_back(spline.x0.flatten(),sample_path_t[0].permute(1,0))

            # Generate samples from rho_t
            
            sample_path_push = spline.push_forward(theta_t[idx].flatten(),z0)
            
            
            # Plot samples
            ax.scatter(sample_path_t[:, 0].detach().cpu().numpy(), 
                       sample_path_t[:, 1].detach().cpu().numpy(), 
                       c='b', alpha=0.25, s=3, label='True path' if idx == 0 else "")
                       
            ax.scatter(sample_path_t2[:, 0].detach().cpu().numpy(), 
                       sample_path_t2[:, 1].detach().cpu().numpy(), 
                       c='r', alpha=0.25, s=3, label='True path approx bd' if idx == 0 else "")
                       
    
            ax.scatter(sample_path_push[:, 0].detach().cpu().numpy(),
                          sample_path_push[:, 1].detach().cpu().numpy(),
                          c='g', alpha=0.25, s=3, label='Approximated path' if idx == 0 else "")

            
            
            if d >= 2:  # Only try to draw ellipses for 2D or higher distributions
                # Extract the first two dimensions for plotting
                mu_t_np = mu_t[i].flatten().detach().cpu().numpy()[:2]
                cov_t_np = cov_t[i].detach().cpu().numpy()[:2, :2]
                
                mu_t2_np = mu_t2[i].flatten().detach().cpu().numpy()[:2]
                cov_t2_np = cov_t2[i].detach().cpu().numpy()[:2, :2]
                
                sol_mean_np = sol_mean[i].flatten().detach().cpu().numpy()[:2]
                sol_cov_np = sol_cov[i].detach().cpu().numpy()[:2, :2]

                ax.plot(mu_t_np[0], mu_t_np[1], 'bo', label='True path mean')
                ax.plot(mu_t2_np[0], mu_t2_np[1], 'ro', label='True path approx bd mean')
                ax.plot(sol_mean_np[0], sol_mean_np[1], 'go', label='Approximated path mean')
                
                # 1. True path (blue)
                
                plot_confidence_ellipse(
                    mean=mu_t_np,
                    cov=cov_t_np,
                    ax=ax,
                    edgecolor='blue',
                    linestyle='-',
                    linewidth=2
                )
                # 3. Approximated path (green)
                
                plot_confidence_ellipse(
                    mean=sol_mean_np,
                    cov=sol_cov_np,
                    ax=ax,
                    edgecolor='green',
                    linestyle=':',
                    linewidth= 4
                )
                
                # 2. True path with approx boundary (red)
                
                plot_confidence_ellipse(
                    mean=mu_t2_np,
                    cov=cov_t2_np,
                    ax=ax,
                    edgecolor='red',
                    linestyle='--',
                    linewidth=2
                )
            
            mean_squared_error = F.mse_loss(sample_path_t.squeeze(0), sample_path_push.permute(1,0))
            ax.set_title(f't = {t[i].item():.2f}, MSE = {mean_squared_error.item():.4f}')

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            # Ensure all subplots have the same scale
            if idx == 0:
                # Get the limits from all distributions
                all_means = np.vstack([
                    mu_t.cpu().numpy(), 
                    mu_t2.cpu().numpy(), 
                    sol_mean.cpu().numpy()
                ])
                all_stds = np.vstack([
                    np.sqrt(np.diagonal(cov_t.cpu().numpy(), axis1=1, axis2=2)),
                    np.sqrt(np.diagonal(cov_t2.cpu().numpy(), axis1=1, axis2=2)),
                    np.sqrt(np.diagonal(sol_cov.cpu().numpy(), axis1=1, axis2=2))
                ])
                all_means = np.reshape(all_means, (-1, 2))
                # Calculate reasonable plot limits
                x_min = np.min(all_means[:, 0] - 3 * np.max(all_stds[:, 0]))
                x_max = np.max(all_means[:, 0] + 3 * np.max(all_stds[:, 0]))
                y_min = np.min(all_means[:, 1] - 3 * np.max(all_stds[:, 1]))
                y_max = np.max(all_means[:, 1] + 3 * np.max(all_stds[:, 1]))
                
                # Store these limits for consistent plotting
                x_lim = (x_min, x_max)
                y_lim = (y_min, y_max)
                
                # Add legend to first plot only
                ax.legend(loc='upper left', fontsize='small')
            
            # Apply consistent limits
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            
            
        plt.tight_layout()
        plt.show()
        
        # Create a figure for error plots
        fig, axes = plt.subplots(3 if l2_errors else 1, 1, figsize=(10, 10 if l2_errors else 4))
        
        # Plot Wasserstein distance error
        if l2_errors:
            ax_w2 = axes[0]
        else:
            ax_w2 = axes
            
        ax_w2.plot(t.detach().cpu().numpy(), w2_error_true_path.detach().cpu().numpy(), label='True path')
        ax_w2.plot(t.detach().cpu().numpy(), w2_error_approx_path.detach().cpu().numpy(), label='Approximated path')
        ax_w2.legend()
        ax_w2.set_title('Wasserstein distance in time')
        ax_w2.grid(True, alpha=0.3)
        
        if l2_errors:
            # Plot errors in mean
            ax_mean = axes[1]
            ax_mean.plot(t.detach().cpu().numpy(), l2_mean_true.detach().cpu().numpy(), label='True path')
            ax_mean.plot(t.detach().cpu().numpy(), l2_mean_approx.detach().cpu().numpy(), label='Approximated path')
            ax_mean.legend()
            ax_mean.set_title('L2 relative error in mean in time')
            ax_mean.grid(True, alpha=0.3)
            
            # Plot errors in covariance
            ax_cov = axes[2]
            ax_cov.plot(t.detach().cpu().numpy(), l2_cov_true.detach().cpu().numpy(), label='True path')
            ax_cov.plot(t.detach().cpu().numpy(), l2_cov_approx.detach().cpu().numpy(), label='Approximated path')
            ax_cov.legend()
            ax_cov.set_title('L2 relative error in covariance in time')
            ax_cov.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        

    # Compute integrals in time
    w2_error_true = torch.trapz(w2_error_true_path.detach().cpu(), t.detach().cpu())
    w2_error_approx = torch.trapz(w2_error_approx_path[1:-2].detach().cpu(), t[1:-2].detach().cpu())
    
    if l2_errors:
        return w2_error_true, w2_error_approx, w2_error_true_path, w2_error_approx_path#, l2_mean_true, l2_cov_true, l2_mean_approx, l2_cov_approx
    
    return w2_error_true, w2_error_approx, w2_error_true_path, w2_error_approx_path
    
        

