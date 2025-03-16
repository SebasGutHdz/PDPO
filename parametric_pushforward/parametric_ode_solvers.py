'''
This code contains the parametric version of the ODE solvers. 
In comparison to the original implementation, the steps pass the arguments to 
the function f. 

We use this feature to be able to assign the weights of the parametric velocity fields.

'''

from torchdyn.numerics.solvers.ode import SolverTemplate
import torch


# Build parametric ODE solvers

class Parametric_Euler(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        """Explicit Euler ODE stepper, order 1"""
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def step(self, f, x, t, dt, k1=None, args=None):
        # Pass args to the function f
        if k1 == None: k1 = f(t, x,*args)
        x_sol = x + dt * k1
        return None, x_sol, None
    
class Parametric_Midpoint(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        """Explicit Midpoint ODE stepper, order 2"""
        super().__init__(order=2)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def step(self, f, x, t, dt, k1=None, args=None):
        # Pass args to the function f
        if k1 == None: k1 = f(t, x,*args)
        x_mid = x + 0.5 * dt * k1
        # Pass args to the function f
        x_sol = x + dt * f(t + 0.5 * dt, x_mid,*args)
        return None, x_sol, None
    


import torchdyn.numerics.solvers.ode as solvers

solvers.Euler = Parametric_Euler
solvers.Midpoint = Parametric_Midpoint

print(f"Successfully patched {solvers.Euler.__name__} and {solvers.Midpoint.__name__}")



