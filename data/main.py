import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    def __init__(self, data, isothermal=True):
        self.Cae = data['Cae']
        self.Cbe = data['Cbe']
        self.Cce = data['Cce']
        self.Cde = data['Cde']
        
        self.k01 = data['k01']
        self.Ea1 = data['Ea1']
        
        self.k02 = data['k02']
        self.Ea2 = data['Ea2']
        
        self.T = data['T']
        self.t_espacial = data['t_espacial']
        self.dHr = data['dHr']
        self.rho = data['rho']
        self.Cp = data['Cp']
        
        self.isothermal = isothermal
    
    def cte_taxa(self, T, k0, Ea, R=8.314):
        """
        k0: [1/min]
        Ea: J/ mol
        R: J/ K. mol
        T: K

        k: [k0]
        """
        return k0*np.exp(-Ea/(R*T))
    
    def ra(self, Ca, Cb, Cc, Cd, T):
        k1 = self.cte_taxa(T, self.k01, self.Ea1)
        k2 = self.cte_taxa(T, self.k02, self.Ea2)
        return -k1*Ca*Cb + k2*Cc*Cd
        
    def dcadt(self, C, t):
        Ca, Cb, Cc, Cd, T = C
        dcdt = [0, 0, 0, 0, 0]
        ra = self.ra(Ca, Cb, Cc, Cd, T)
        
        dcdt[0] = (self.Cae-Ca)/self.t_espacial + ra
        dcdt[1] = (self.Cbe-Cb)/self.t_espacial + ra
        dcdt[2] = (self.Cce-Cc)/self.t_espacial - ra
        dcdt[3] = (self.Cde-Cd)/self.t_espacial - ra
        if self.isothermal:
            dcdt[4] = 0
        else:
            dcdt[4] = (self.T - T)/self.t_espacial - self.dHr*ra/(self.rho* self.Cp)
        return dcdt


def ss_solve(data, isothermal=True):
    """
    Ce (nx4): [Cae, Cbe, Cce, Cde]
    k1 (nx1):
    k2 (nx1):
    Q  (nx1):
    V       :
    """
    out = []
    for row in range( data.shape[0] ):
        s = steadyState(data.loc[row], isothermal)
        out.append(fsolve(s.dcadt,
                         [s.Cae, s.Cbe, s.Cce, s.Cde, s.T]))
    return pd.DataFrame(out, columns=['Ca', 'Cb', 'Cc', 'Cd', 'T'])

def ode_solve(data, n, t=10, row=0, isothermal=True):
    """
    Solve 1 ODE
    """
    t = np.linspace(0, t, n)
    s = steadyState(data.loc[row], isothermal)
    out = pd.DataFrame(odeint(s.dcadt,
                              [s.Cae, s.Cbe, s.Cce, s.Cde, s.T],
                              t),
                       columns=['Ca', 'Cb', 'Cc', 'Cd', 'T'], index=t)
    out = out[out>=0].dropna()
    return out

def ode_plot(ode_solved):
    """
    
    """
    plt.figure(figsize=(11,6))
    axis_x = ode_solved.index/60
    ax1 = plt.plot(axis_x, ode_solved[['Ca', 'Cb', 'Cc', 'Cd']]*100/ode_solved['Ca'][0], label=['Ca', 'Cb', 'Cc', 'Cd'], linewidth=2)
    plt.xlabel('tempo [min]')
    plt.ylabel('%')
    plt.grid()
    
    # Temperature axis
    ax2 = plt.twinx()
    ax2 = ax2.plot(axis_x, ode_solved[['T']], label='T', color='black', linewidth=3)
    
    # complements
    plt.xlabel('tempo [s]')
    plt.ylabel('T [K]')

    ax = ax1 + ax2
    legends = [l.get_label() for l in ax]
    plt.legend(ax, legends, loc='best', ncol=5, )

    plt.show()