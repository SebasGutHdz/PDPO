'''
Reference:
https://github.com/facebookresearch/generalized-schrodinger-bridge-matching/blob/main/gsbm/state_cost.py#L455
'''

import torch
import torch.nn.functional as F
import numpy as np



def obstacle_cfg_stunnel():
    a, b, c = 20, 1, 90
    centers = [[5, 6], [-5, -6]]
    return a, b, c, centers

def obstacle_cfg_gmm():
    centers = [[6, 6], [6, -6], [-6, -6]] # it was 3  [3.5 , 3.5], [3.5, -3.5], [-3.5, -3.5]
    radius = 1.5 # it was .75
    return centers, radius


def obstacle_cfg_stunnel():
    a, b, c = 20, 1, 90
    centers = [[5, 6], [-5, -6]]
    return a, b, c, centers
def obstacle_cfg_vneck():
    c_sq = 0.36
    coef = 2
    return c_sq, coef


def obstacle_cost_gmm(xt):

    Bs, D = xt.shape[:-1], xt.shape[-1]
    assert D == 2
    xt = xt.reshape(-1, xt.shape[-1])

    batch_xt = xt.shape[0]

    centers, radius = obstacle_cfg_gmm()

    obs1 = torch.tensor(centers[0]).repeat((batch_xt, 1)).to(xt.device)
    obs2 = torch.tensor(centers[1]).repeat((batch_xt, 1)).to(xt.device)
    obs3 = torch.tensor(centers[2]).repeat((batch_xt, 1)).to(xt.device)

    dist1 = torch.norm(xt - obs1, dim=-1)
    dist2 = torch.norm(xt - obs2, dim=-1)
    dist3 = torch.norm(xt - obs3, dim=-1)

    cost1 = F.softplus(100*(radius - dist1), beta=1, threshold=20)
    cost2 = F.softplus(100*(radius - dist2), beta=1, threshold=20)
    cost3 = F.softplus(100*(radius - dist3), beta=1, threshold=20)
    return (cost1 + cost2 + cost3).reshape(*Bs)*50


# def obstacle_cost_stunnel(xt):
#     """
#     xt: (*, 2) -> (*,)
#     """

#     a, b, c, centers = obstacle_cfg_stunnel()

#     Bs, D = xt.shape[:-1], xt.shape[-1]
#     assert D == 2

#     _xt = xt.reshape(-1, D)
#     x, y = _xt[:, 0], _xt[:, 1]

#     d = a * (x - centers[0][0]) ** 2 + b * (y - centers[0][1]) ** 2
#     # c1 = 1500 * (d < c)
#     c1 = F.softplus(c - d, beta=1, threshold=20)

#     d = a * (x - centers[1][0]) ** 2 + b * (y - centers[1][1]) ** 2
#     # c2 = 1500 * (d < c)
#     c2 = F.softplus(c - d, beta=1, threshold=20)

#     cost = (c1 + c2).reshape(*Bs)
#     return cost*1500


def obstacle_cost_vneck(xt):
    """
    xt: (*, 2) -> (*,)
    """
    assert xt.shape[-1] == 2

    c_sq, coef = obstacle_cfg_vneck()

    xt_sq = torch.square(xt)
    d = coef * xt_sq[..., 0] - xt_sq[..., 1]

    return F.softplus(-c_sq - d, beta=1, threshold=20)*3000
    # return 15000 * (d < -c_sq)

def congestion_cost(xt):
    '''
    xt: (Bs,T,d) --> (Bs,T)
    '''
    T,d = xt.shape[-2:]

    yt = xt.reshape(-1,T,d)
    yt = yt[torch.randperm(yt.shape[0])].reshape(xt.shape)

    dd = xt-yt
    dist = torch.sum(dd**2,dim = -1)

    congestion = 2.0/(dist+1e0)

    assert congestion.shape == xt.shape[:-1]

    return congestion*50

def quadartic_well(xt):
    '''
    xt: (Bs,T,d) --> (Bs,T)
    '''
    T, d = xt.shape[-2:]
    
    # Increased coefficient matrix (5x stronger than original)
    A = torch.tensor([[2.5, 0], [0, 2.5]]).to(xt.device)
    
    # Base quadratic potential
    base_potential = -torch.sum((xt @ A) * xt, dim=-1) * .5
    
    # Calculate distance from center
    center_dist = torch.sum(xt**2, dim=-1)
    
    # Apply exponential sharpening near center
    # This creates steeper slopes near the origin without changing sign
    sharpening_factor = torch.exp(-center_dist * 0.5) + 1.0
    
    # Final potential with enhanced center sharpness
    sharp_potential = base_potential * sharpening_factor
    
    return sharp_potential
    

    

def geodesic(xt):
    '''
    xt: (Bs,T,d) --> (Bs,T)
    '''
    T,d = xt.shape[-2:]

    return torch.zeros((xt.shape[0],T)).to(xt.device)


def obstacle_cost_stunnel(xx_inp,scale=1):

    """
    Calculate interaction term. Calculates F(x), where F is the forcing term in the HJB equation.
    xx_inp: (bs,t,d) -> (bs,t)
    """
    
    device = xx_inp.device
    xx = xx_inp[:,:, 0:2]
    xx = xx.view(-1, 2)  # Flatten the input to (bs*t, d)
    batch_size = xx.size(0)
    dim = xx.size(1)
    # assert (dim == 2), f"Require dim=2 but, got dim={dim} (BAD)"

    # Two diagonal obstacles
    # Rotation matrix
    theta = torch.tensor(np.pi / 5)
    rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]]).expand(batch_size, dim, dim).to(device)

    # Bottom/Left obstacle  # TODO: Clean it up
    center1 = torch.tensor([-2, 0.5], dtype=torch.float).to(device)
    xxcent1 = xx - center1
    xxcent1 = xxcent1.unsqueeze(1).bmm(rot_mat).squeeze(1)
    covar_mat1 = torch.eye(dim, dtype=torch.float)
    covar_mat1[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
    covar_mat1 = covar_mat1.expand(batch_size, dim, dim).to(device)
    bb_vec1 = torch.tensor([0, 2], dtype=torch.float).expand(xx.size()).to(device)
    xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
    quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
    lin1 = torch.sum(xxcent1 * bb_vec1, dim=1, keepdim=True)
    out1 = (-1) * ((quad1 + lin1) + 1)
    out1 = scale * out1.view(-1, 1)
    out1 = torch.clamp_min(out1, min=0)

    # Top/Right obstacle
    center2 = torch.tensor([2, -0.5], dtype=torch.float).to(device)
    xxcent2 = xx - center2
    xxcent2 = xxcent2.unsqueeze(1).bmm(rot_mat).squeeze(1)
    covar_mat2 = torch.eye(dim, dtype=torch.float)
    covar_mat2[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
    covar_mat2 = covar_mat2.expand(batch_size, dim, dim).to(device)
    bb_vec2 = torch.tensor([0, -2], dtype=torch.float).expand(xx.size()).to(device)
    xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat2)
    quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
    lin2 = torch.sum(xxcent2 * bb_vec2, dim=1, keepdim=True)
    out2 = (-1) * ((quad2 + lin2) + 1)
    out2 = scale * out2.view(-1, 1)
    out2 = torch.clamp_min(out2, min=0)

    out = out1 + out2

    out = out.view(batch_size, -1)
    out = out.view(xx_inp.size(0), xx_inp.size(1))

    return out*2000