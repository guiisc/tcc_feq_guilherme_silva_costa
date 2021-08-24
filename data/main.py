import numpy as np
np.random.seed(0)
import pandas as pd

R = 8.314 # J/mol.K
k0 = .311 # 1/min
Ea = 0
Kc = 1e12
V = 1

def cte_taxa(T):
    """
    """
    return k0*np.exp(-Ea/(R*T))

def Ci(Ca, Cb, Cc, Cd, Q, T):
    """
    """
    return (cte_taxa(T)*V/Q) * (Ca*Cb - Cc*Cd/Kc)

def generate_values(Vmin, Vmax, n):
    """
    Tmin and Tmax of a range 
    n: number of data
    """
    V_range = np.linspace(Vmin, Vmax, int(.001*n)+2)
    f_V_range = lambda x: V_range[x]
    
    V = (np.random.rand(n) - 0.5)*(Vmax - Vmin)/10
    V += f_V_range( np.random.randint(0, V_range.shape[0], n) )
    return V