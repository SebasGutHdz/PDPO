import torch as torch
import numpy as np

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
    
        

