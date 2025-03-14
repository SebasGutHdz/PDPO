import torch as torch
import torch.nn.functional as F


def linear(t,xt,s):
    '''
    https://github.com/facebookresearch/generalized-schrodinger-bridge-matching
    t: (T,1)
    xt: (T,1,D)
    s: (S,1)
    '''

    T,N,D = xt.shape
    S = s.shape[0]

    m = (xt[1:]-xt[:-1])/(t[1:]-t[:-1] +1e-10).unsqueeze(-1)
    
    left = torch.searchsorted(t[1:].T.contiguous(),s.T.contiguous(), side='left').T
    mask_l = F.one_hot(left,T).permute(0,2,1).reshape(S,T,N,1)

    t = t.reshape(1,T,N,1)
    xt = xt.reshape(1,T,N,D)
    m = m.reshape(1,T-1,N,D)
    s = s.reshape(S,N,1)

    x0 = torch.sum(t*mask_l,dim = 1)
    p0 = torch.sum(xt*mask_l,dim = 1)
    m0 = torch.sum(m*mask_l[:,:-1],dim = 1)

    t = s-x0

    return t*m0+p0

def cubic_interp(t,xt,s):
    '''
    # Functions obtained from:https://github.com/facebookresearch/generalized-schrodinger-bridge-matching
    t: (T,1)
    xt: (T,1,D)
    s: (S,1)
    '''
    T,N,D = xt.shape
    S = s.shape[0]

    if t.shape == s.shape:
        return xt

    # Compute derivative by centered difference

    fwd = (xt[1:]-xt[:-1])/(t[1:]-t[:-1] +1e-10).unsqueeze(-1)
    # Derivatives for interior points
    m = (fwd[1:]+fwd[:-1])/2
    # Derivative for the first point
    m = torch.cat([fwd[0:1],m],dim = 0)
    # Derivative for the last point
    m = torch.cat([m,fwd[-1:]],dim = 0)

    # Compute the left index for each query point
    left = torch.searchsorted(t[1:].T.contiguous(),s.T.contiguous(), side='left').T
    mask_l = F.one_hot(left,T).permute(0,2,1).reshape(S,T,N,1)
    right = (left + 1) 
    mask_r = F.one_hot(right,T).permute(0,2,1).reshape(S,T,N,1)

    t = t.reshape(1,T,N,1)
    xt = xt.reshape(1,T,N,D)
    m = m.reshape(1,T,N,D)
    s = s.reshape(S,N,1)

    x0 = torch.sum(t*mask_l,dim = 1)
    x1 = torch.sum(t*mask_r,dim = 1)
    p0 = torch.sum(xt*mask_l,dim = 1)
    p1 = torch.sum(xt*mask_r,dim = 1)
    m0 = torch.sum(m*mask_l,dim = 1)
    m1 = torch.sum(m*mask_r,dim = 1)

    t = s-x0
    t = t/(x1-x0+1e-10)

    return (2*t**3-3*t**2+1)*p0 + (t**3-2*t**2+t)*m0*(x1-x0) + (-2*t**3+3*t**2)*p1 + (t**3-t**2)*m1*(x1-x0)



def dervi_cubic_interp(t,xt,s):
    
    '''
    # Functions obtained from:https://github.com/facebookresearch/generalized-schrodinger-bridge-matching
    t: (T,1)
    xt: (T,1,D)
    s: (S,1)
    '''
    T,N,D = xt.shape
    S = s.shape[0]

    

    # Compute derivative by centered difference

    fwd = (xt[1:]-xt[:-1])/(t[1:]-t[:-1] +1e-10).unsqueeze(-1)
    # Derivatives for interior points
    m = (fwd[1:]+fwd[:-1])/2
    # Derivative for the first point
    m = torch.cat([fwd[0:1],m],dim = 0)
    # Derivative for the last point
    m = torch.cat([m,fwd[-1:]],dim = 0)

    if t.shape == s.shape:
        return m

    # Compute the left index for each query point
    left = torch.searchsorted(t[1:].T.contiguous(),s.T.contiguous(), side='left').T
    mask_l = F.one_hot(left,T).permute(0,2,1).reshape(S,T,N,1)
    right = (left + 1) 
    mask_r = F.one_hot(right,T).permute(0,2,1).reshape(S,T,N,1)

    t = t.reshape(1,T,N,1)
    xt = xt.reshape(1,T,N,D)
    m = m.reshape(1,T,N,D)
    s = s.reshape(S,N,1)

    x0 = torch.sum(t*mask_l,dim = 1)
    x1 = torch.sum(t*mask_r,dim = 1)
    p0 = torch.sum(xt*mask_l,dim = 1)
    p1 = torch.sum(xt*mask_r,dim = 1)
    m0 = torch.sum(m*mask_l,dim = 1)
    m1 = torch.sum(m*mask_r,dim = 1)

    t = s-x0
    t = t/(x1-x0+1e-10)

    return ((6*t**2-6*t)*p0/(x1-x0+1e-10)  + (3*t**2-4*t+1)*m0 + (-6*t**2+6*t)*p1/(x1-x0+1e-10)  + (3*t**2-2*t)*m1)

