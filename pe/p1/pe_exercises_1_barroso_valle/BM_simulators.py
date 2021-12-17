# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: Alberto Suárez
"""
# Load packages
import numpy as np


def simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N):
    """ Simulation in [t0,t0+T] of arithmetic Brownian motion trajectories

        SDE:    dB(t) = mu*dt + sigma*dW(t)
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    B0 : float
        Initial level of the process
    T : float
        Length of the simulation 
    mu, sigma : float
        Parameters of the process
    M : int
        Number of trajectories in simulation
    N : int
        Number of steps for the simulation
            
    Returns
    -------
    t: numpy.ndarray [float] of shape (N+1,)
        Regular grid of discretization times in [t0,t0+T]
    B: numpy.ndarray [float] of shape (M, N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values of the process at t 
        

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import BM_simulators as BM
    >>> t0, B0, T, mu, sigma = 0, 10.0, 2.0, 1.5, 0.4
    >>> M, N = 20, 1000
    >>> t, B = BM.simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> _ = plt.plot(t,B.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('B(t)') 
    >>> _= plt.title('Arithmetic Brownian motion in 1D')

    """
    
    dT = T / N    # integration step
    t = np.linspace(t0, t0+T, N+1) # integration grid
    Z = np.random.randn(M, N)    # Gausssian White noise
    d = mu*dT + sigma*np.sqrt(dT)*Z  # additive factor
    B = np.concatenate((B0*np.ones((M, 1)), d), axis=1)
    B = np.cumsum(B, axis=1)
    return t, B

def simulate_geometric_BM(t0, S0, T, mu, sigma, M, N):
    """ Simulation in [t0,t0+T] of geometric Brownian motion trajectories

        SDE:    dS(t) = mu*S(t)*dt + sigma*S(t)*dW(t)
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    S0 : float
        Initial level of the process
    T : float
        Length of the simulation 
    mu, sigma : float
        Parameters of the process
    M: int
        Number of trajectories in simulation
    N: int
        Number of steps for the simulation
            
    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t0+T]
    S: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values of the process at t 

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import BM_simulators as BM
    >>> t0, S0, T, mu, sigma = 0, 10.0, 2.0, 0.3, 0.4
    >>> M, N = 20, 1000
    >>> t, S = BM.simulate_geometric_BM(t0, S0, T, mu, sigma, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)') 
    >>> _= plt.title('Geometric Brownian motion in 1D')

    """
    
    dT = T / N    # integration step
    t = np.linspace(t0, t0+T, N+1) # integration grid
    Z = np.random.randn(M, N)  # Gausssian White noise
    e = np.exp((mu-0.5*sigma**2)*dT + sigma*np.sqrt(dT)*Z) # multiplicative factor
    S = np.concatenate((S0*np.ones((M, 1)), e), axis=1)
    S = np.cumprod(S, axis=1)
    return t, S

def simulate_Brownian_bridge(t0, B0, t1, B1, sigma, M, N):
    """ Simulation in [t0,t1] of Brownian bridge trajectories
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    B0 : float
        Initial level of the process
    t1 : float
        Final time for the simulation
    B1 : float
        Final level of the process
    sigma : float
        Parameter of the process
    M: int
        Number of trajectories in simulation
    N: int
        Number of steps for the simulation
            
    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    BB: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values of the process at t 

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import BM_simulators as BM
    >>> t0, B0, t1, B1, sigma = 0, 10.0, 2.0, 12.0, 0.4
    >>> M, N = 20, 1000
    >>> t, BB = BM.simulate_Brownian_bridge(t0, B0, t1, B1, sigma, M, N)
    >>> _ = plt.plot(t,BB.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('BB(t)') 
    >>> _= plt.title('Brownian bridge')
    
    """       
    
    t, B = simulate_arithmetic_BM(t0, B0, t1-t0, 0, sigma, M, N)
    B_t1 = B[:,-1]
    BB = B + np.multiply.outer(B1 - B_t1, (t-t0) / (t1-t0))
    return t, BB

def simulate_Ornstein_Uhlenbeck(t0, X0, T, k, D, M, N):
    """ Simulation in [t0, t0+T] of Ornstein-Uhlenbeck process trajectories
    
        SDE:    dX(t) = - k*X(t)*dt + sqrt(D)*dW(t)
    
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    X0 : float
        Initial level of the process
    T : float
        Length of the simulation 
    k : float
        Rate or inverse correlation time of the process
    D : float
        Diffusion constant
    M: int
        Number of trajectories in simulation
    N: int
        Number of steps for the simulation
            
    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values of the process at t
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import BM_simulators as BM
    >>> t0, X0, T, k, D = [0.0, 10.0, 20.0, 0.5, 0.4]
    >>> M, N = 20, 1000
    >>> t, X = BM.simulate_Ornstein_Uhlenbeck(t0, X0, T, k, D, M, N)
    >>> _ = plt.plot(t,X.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('X(t)') 
    >>> _= plt.title('Ornstein-Uhlenbeck process')
    
    """

    dT = T / N    # integration step
    t = np.linspace(t0, t0+T, N+1) # integration grid
    
    Z = np.random.randn(M, N)    # Gausssian White noise
    
    X = np.zeros((M,N+1))
    X[:,0] = X0

    sigma = np.sqrt(D / k * (1.0 - np.exp(-2.0*k*dT)))

    for n in np.arange(N):
        X[:,n+1] = X[:,n]*np.exp(-k*dT) + sigma*Z[:,n]
    return t, X
