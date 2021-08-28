import numpy as np
np.random.seed(0)
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import fsolve


def cte_taxa(T, k0, Ea, R=8.314):
    """
    k0: [1/min]
    Ea: J/ mol
    R: J/ K. mol
    T: K
    
    k: [k0]
    """
    return k0*np.exp(-Ea/(R*T))

    
def X(Ce, Cs):
    return 1 - Cs/Ce

def generate_values(Vmin, Vmax, n):
    """
    Tmin and Tmax of a range 
    n: number of data
    """
    V_range = np.linspace(Vmin, Vmax, int(.005*n)+2)
    f_V_range = lambda x: V_range[x]
    
    V = (np.random.rand(n) - 0.5)*(Vmax - Vmin)/10
    V += f_V_range( np.random.randint(0, V_range.shape[0], n) )
    return abs(V)

class steadyState:
    def __init__(self, Ce, k1, k2, Q, V):
        self.Ce = Ce
        self.Cae = Ce[0]
        self.Cbe = Ce[1]
        self.Cce = Ce[2]
        self.Cde = Ce[3]
        self.k1 = k1
        self.k2 = k2
        self.Q = Q
        self.V = V
    
    def ra(self, Ca, Cb, Cc, Cd):
        return -self.k1*Ca*Cb + self.k2*Cc*Cd
        
    def dcadt(self, C):
        Ca, Cb, Cc, Cd = C
        dcdt = [0, 0, 0, 0]
        dcdt[0] = self.Q*(self.Cae-Ca)/self.V + self.ra(Ca, Cb, Cc, Cd)
        dcdt[1] = self.Q*(self.Cbe-Cb)/self.V + self.ra(Ca, Cb, Cc, Cd)
        dcdt[2] = self.Q*(self.Cce-Cc)/self.V - self.ra(Ca, Cb, Cc, Cd)
        dcdt[3] = self.Q*(self.Cde-Cd)/self.V - self.ra(Ca, Cb, Cc, Cd)
        return dcdt


def ss_solve(Ce, k1, k2, Q, V):
    """
    Ce (nx4): [Cae, Cbe, Cce, Cde]
    k1 (nx1):
    k2 (nx1):
    Q  (nx1):
    V       :
    """
    out = []
    for row in range( Ce.shape[0] ):
        s = steadyState(Ce.loc[row].values, k1[row], k2[row], Q[row], V)
        out.append(fsolve(s.dcadt, s.Ce))
    return pd.DataFrame(out, columns=['Ca', 'Cb', 'Cc', 'Cd'])
        


"""------------------------------------------------------------------------"""
"""------------------------------------------------------------------------"""
"""------------------------------------------------------------------------"""
class data:
    def __init__(self, Ce, k1, k2, Q, V):
        self.Ce = Ce
        self.Cae = Ce[0]
        self.Cbe = Ce[1]
        self.Cce = Ce[2]
        self.Cde = Ce[3]
        self.k1 = k1
        self.k2 = k2
        self.Q = Q
        self.V = V
        
    def dcadt(self, C, t):
        Ca, Cb, Cc, Cd = C
        dcdt = [0,0,0,0]
        dcdt[0] = self.Q*(Ca-self.Cae)/self.V - self.k1*Ca*Cb + self.k2*Cc*Cd
        dcdt[1] = self.Q*(Cb-self.Cbe)/self.V - self.k1*Ca*Cb + self.k2*Cc*Cd
        dcdt[2] = self.Q*(Cc-self.Cce)/self.V + self.k1*Ca*Cb - self.k2*Cc*Cd
        dcdt[3] = self.Q*(Cd-self.Cde)/self.V + self.k1*Ca*Cb - self.k2*Cc*Cd
        return dcdt
    
def solve(C, n, k1, k2, Q, V):
    t = np.linspace(0, 10, n)
    d = data(C, k1, k2, Q, V)
    out = pd.DataFrame(odeint(d.dcadt, d.Ce, t), columns=['Ca', 'Cb', 'Cc', 'Cd'], index=t)
    out = out[out>=0].dropna()
    return out

def gen_(Ca, Cb, Cc, Cd, Q, V, k1, k2):
    n = Ca.shape[0]
    out = []
    for i in range(n):
        C = [Ca[i], Cb[i], Cc[i], Cd[i]]
        out_aux = solve(C, n, k1[i], k2[i], Q[i], V)
        try:
            out.append(out_aux.iloc[-1].values)
        except:
            print(i)
            return out_aux
    
    return np.array(out)

def gen(data, V):
    Ca = data['Cae']
    Cb = data['Cbe']
    Cc = data['Cce']
    Cd = data['Cde']
    Q = data['Q']
    k1 = data['k1']
    k2 = data['k2']
    
    n = data.shape[0]
    out = []
    for i in range(n):
        C = [Ca[i], Cb[i], Cc[i], Cd[i]]
        out_aux = solve(C, n, k1[i], k2[i], Q[i], V)
        try:
            out.append(out_aux.iloc[-1].values)
        except:
            print(i)
            return out_aux
    
    return pd.DataFrame(np.array(out), columns=['Ca', 'Cb', 'Cc', 'Cd'])