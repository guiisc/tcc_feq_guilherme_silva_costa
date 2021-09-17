import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

    
def X(Ce, Cs):
    """
    Return the reaction conversion
    
    Parameters:
    -----------
    :param Ce (int/numpy.array): Initial concentration
    :param Cs (int/numpy.array): Final concentration
    
    :return (float): Conversion
    """
    return 1 - Cs/Ce

def generate_values(Vmin, Vmax, n):
    """
    Generates n random values from a range defined by Vmin and Vmax
    
    Parameters:
    -----------
    :param Vmin (float): minimum side of the range
    :param Vmax (float): maximum side of the range
    :param n (int): number of data
    
    :return (numpy.array): Random values of that range
    """
    V_range = np.linspace(Vmin, Vmax, int(.005*n)+2)
    f_V_range = lambda x: V_range[x]
    
    V = (np.random.rand(n) - 0.5)*(Vmax - Vmin)/10
    V += f_V_range( np.random.randint(0, V_range.shape[0], n) )
    return abs(V)

class steadyState:
    """
    Define a EDO system that can be used to evaluate the steady state as well the reactor startup
    """
    def __init__(self, data, isothermal=True):
        """
        Instanciate a class with the initial data as well the properties data.
        
        Parameters:
        -----------
        :param data (pandas.DataFrame): Initial points of concentrations, temperature, constante rate, spacial time and thermal properties [Cae, Cbe, Cce, Cde, T, k01, Ea1, k02, Ea2, t_spacial, dHr, rho, Cp]
        :param isothermal (boolean): If the ractor will be operated in a isothermal (True) or adiabatic state (False).
        """
        self.Cae = data['Cae']
        self.Cbe = data['Cbe']
        self.Cce = data['Cce']
        self.Cde = data['Cde']
        
        self.k01 = data['k01']
        self.Ea1 = data['Ea1']
        
        self.k02 = data['k02']
        self.Ea2 = data['Ea2']
        
        self.T = data['T']
        self.t_spacial = data['t_spacial']
        self.dHr = data['dHr']
        self.rho = data['rho']
        self.Cp = data['Cp']
        
        self.isothermal = isothermal
    
    def constant_rate(self, T, k0, Ea, R=8.314):
        """
        Evaluate the constant rate
        
        Parameters:
        -----------
        :param k0 (float): Pre-exponential factor [1/min]
        :param Ea (float): Activation energy [J/ mol]
        :param R (float): Ideal gas constant [J/ K.mol]
        :param T (float): Temperatura [K]

        :return (float):
        """
        return k0*np.exp(-Ea/(R*T))
    
    def ra(self, Ca, Cb, Cc, Cd, T):
        """
        Evaluate the reaction rate
        
        Parameters:
        -----------
        :param Ca (float): Concentration of component A
        :param Cb (float): Concentration of component B
        :param Cc (float): Concentration of component C
        :param Cd (float): Concentration of component D
        :param T (float): Temperature
        """
        k1 = self.constant_rate(T, self.k01, self.Ea1)
        k2 = self.constant_rate(T, self.k02, self.Ea2)
        return -k1 + k2
        return -k1*Ca*Cb + k2*Cc*Cd
        
    def dcadt(self, C, t):
        """
        Used to evaluate the EDO system at time t, using data from time t-1
        
        Parameters:
        -----------
        :param C (numpy.array): Array with values of concentration and temperature at t-1
        :param t (numpy.array): Array with the time array which the EDO system are being evaluated
        """
        Ca, Cb, Cc, Cd, T = C
        dcdt = [0, 0, 0, 0, 0]
        ra = self.ra(Ca, Cb, Cc, Cd, T)
        
        dcdt[0] = (self.Cae-Ca)/self.t_spacial + ra
        dcdt[1] = (self.Cbe-Cb)/self.t_spacial + ra
        dcdt[2] = (self.Cce-Cc)/self.t_spacial - ra
        dcdt[3] = (self.Cde-Cd)/self.t_spacial - ra
        if self.isothermal:
            dcdt[4] = 0
        else:
            dcdt[4] = (self.T - T)/self.t_spacial - self.dHr*ra/(self.rho* self.Cp)
        return dcdt


def ss_solve(data, isothermal=True):
    """
    Solve the ODE system for the steady state, for all initial points in data
    
    Parameters:
    -----------
    :param data (pandas.DataFrame): Initial points of concentrations, temperature, constante rate, spacial time and thermal properties [Cae, Cbe, Cce, Cde, T, k01, Ea1, k02, Ea2, t_spacial, dHr, rho, Cp]
    :param isothermal (boolean): If the ractor will be operated in a isothermal (True) or adiabatic state (False).
    """
    out = []
    for row in range( data.shape[0] ):
        s = steadyState(data.loc[row], isothermal)
        out.append(fsolve(s.dcadt,
                         [s.Cae, s.Cbe, s.Cce, s.Cde, s.T], args=(1,)))
    return pd.DataFrame(out, columns=['Ca', 'Cb', 'Cc', 'Cd', 'T'])

def ode_solve(data, n, t=10, row=0, isothermal=True):
    """
    Solve a ODE system based on initial values in data
    
    Parameters:
    -----------
    :param data (pandas.DataFrame): 
    :param n (int): Number of points
    :param t (float): Maximum time, in seconds, which the system will be evaluated
    :param row (int): Which row of data will be evaluated
    :param isothermal (boolean): If the ractor will be operated in a isothermal (True) or adiabatic state (False).
    
    :return (pandas.DataFrame): Solution through time of the system.
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
    Ploting method of the startup of a reactor
    
    Parameters:
    -----------
    :param ode_solved (pandas.DataFrame): Solution of a the ODE system
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