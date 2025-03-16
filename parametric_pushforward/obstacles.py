'''
Reference:
https://github.com/facebookresearch/generalized-schrodinger-bridge-matching/blob/main/gsbm/state_cost.py#L455
'''

import torch
import torch.nn.functional as F



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

    cost1 = F.softplus(1*(radius - dist1), beta=1, threshold=20)
    cost2 = F.softplus(1*(radius - dist2), beta=1, threshold=20)
    cost3 = F.softplus(1*(radius - dist3), beta=1, threshold=20)
    return (cost1 + cost2 + cost3).reshape(*Bs)*20_000


def obstacle_cost_stunnel(xt):
    """
    xt: (*, 2) -> (*,)
    """

    a, b, c, centers = obstacle_cfg_stunnel()

    Bs, D = xt.shape[:-1], xt.shape[-1]
    assert D == 2

    _xt = xt.reshape(-1, D)
    x, y = _xt[:, 0], _xt[:, 1]

    d = a * (x - centers[0][0]) ** 2 + b * (y - centers[0][1]) ** 2
    # c1 = 1500 * (d < c)
    c1 = F.softplus(c - d, beta=1, threshold=20)

    d = a * (x - centers[1][0]) ** 2 + b * (y - centers[1][1]) ** 2
    # c2 = 1500 * (d < c)
    c2 = F.softplus(c - d, beta=1, threshold=20)

    cost = (c1 + c2).reshape(*Bs)
    return cost*1000


def obstacle_cost_vneck(xt):
    """
    xt: (*, 2) -> (*,)
    """
    assert xt.shape[-1] == 2

    c_sq, coef = obstacle_cfg_vneck()

    xt_sq = torch.square(xt)
    d = coef * xt_sq[..., 0] - xt_sq[..., 1]

    return F.softplus(-c_sq - d, beta=1, threshold=20)*1000
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

    return congestion*100

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