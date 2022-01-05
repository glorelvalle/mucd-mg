# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: Alberto Suárez
"""
# Load packages
import numpy as np


def euler_maruyana(t0, x0, T, a, b, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    b :
        Function b(t,x(t)) that characterizes the diffusion term
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values
        of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> t, S = sde.euler_maruyana(t0, S0, T, a, b, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Euler scheme)')

    """
    dT = T / N # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0+T, N+1) # integration grid

    X = np.zeros((M, N+1))

    # Initial condition
    X[:, 0] = x0

    # Integration of the SDE
    for n in range(N):
        dW = np.random.randn(M)
        X[:, n+1] = X[:, n] + a(t[n], X[:, n])*dT + b(t[n], X[:, n])*np.sqrt(dT)*dW

    return t, X



def milstein(t0, x0, T, a, b, db_dx, M, N):
    """ Numerical integration of an SDE using the stochastic Milstein scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t, x(t)) that characterizes the drift term
    b :
        Function b(t, x(t)) that characterizes the diffusion term
    db_dx:
        Derivative wrt the second argument of b(t, x)
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> def db_dSt(t, St): return sigma
    >>> t, S = sde.milstein(t0, S0, T, a, b, db_dSt, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Milstein scheme)')

    """
    dT = T/N # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0+T, N+1) # integration grid

    X = np.zeros((M, N+1))

    # Initial condition
    X[:, 0] = x0

    # Integration of the SDE
    for n in range(N):
        dW = np.random.randn(M)
        X[:, n+1] =  X[:, n] + a(t[n], X[:, n])*dT + b(t[n], X[:, n])*np.sqrt(dT)*dW + 0.5*db_dx(t[n], X[:, n])*b(t[n], X[:, n])*(dW**2 - 1)*dT

    return t, X

    

def simulate_jump_process(t0, T, simulator_arrival_times, simulator_jumps, M):
    """ Simulation of jump process

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : float
        Length of the simulation interval [t0, t0+T]
    simulator_arrival_times: callable with arguments (t0,T)
        Function that returns a list of M arrays of arrival times in [t0, t0+T]
    simulator_jumps: callable with argument N
        Function that returns a list of M arrays with the sizes of the jumps
    M: int
        Number of trajectories in the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    """

    times_of_jumps = [[] for _ in range(M)]
    sizes_of_jumps = [[] for _ in range(M)]
    for m in range(M):
        times_of_jumps[m] = simulator_arrival_times(t0, T)
        max_jumps = len(times_of_jumps[m])
        sizes_of_jumps[m] = simulator_jumps(max_jumps)
    return times_of_jumps, sizes_of_jumps


# Stochastic Euler scheme for the numerical solution of a jump-diffision SDE
def euler_jump_diffusion(t0, x0, T, a, b, c,
                         simulator_jump_process,
                         M, N):
    """ Simulation of jump diffusion process

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t) + c(t, x(t)) dJ(t)

    [Itô SDE with a jump term]


    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a : Function a(t,x(t)) that characterizes the drift term
    b : Function b(t,x(t)) that characterizes the diffusion term
    c : Function c(t,x(t)) that characterizes the jump term
    simulator_jump_process: Function that returns times and sizes of jumps
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t
    """
    # Init
    dT = T / N                                                                      # integration step
    t = np.linspace(t0, t0+T, N+1)                                                  # integration grid
    Z = np.random.randn(M, N)                                                       # gaussian white noise
    X = x0*np.ones((M, N+1))                                                        # init trajectories simulation
    times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)               # jumps and sizes times
    indexes = list(map(lambda ts: (N*(ts - t0)/T).astype(int), times_of_jumps))     # closest represented times indexes

    # Compute jump-diffusion SDE
    for m in range(M):
        for n in range(1, N+1):
            a_b = X[m, n-1] + a(t[n-1], X[m, n-1])*dT + b(t[n-1], X[m, n-1])*np.sqrt(dT)*Z[m, n-1]
            actual_time = np.where(indexes[m]==n)[0]
            if len(actual_time) > 0:
                X[m, n] = a_b + c(t[n-1], X[m, n-1])*sizes_of_jumps[m][actual_time[0]]
            else:
                X[m, n] = a_b
    return t, X
