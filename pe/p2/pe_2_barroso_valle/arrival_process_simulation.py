# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:08:31 2020

@author: alberto
"""
# Load packages
import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad

def simulate_poisson(t0, t1, lambda_rate, M):
    """ Simulation of homogeneours Poisson process in [t0, t1]
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    t1 : float
        Final time for the simulation
    M: int
        Number of count sequences in simulation
            
    Returns
    -------
    times: list of M lists 
        Simulation consisting of M sequences (lists) of arrival times.
 
    Examples
    -------
    >>> import matplotlib.pyplot as plt
    >>> import arrival_process_simulation as arrival
    >>> import numpy as np
    >>> lambda_rate = 0.5
    >>> t0 = 0.0
    >>> t1 = 200.0
    >>> M = 3 
    >>> arrival_times = arrival.simulate_poisson(t0, T, lambda_rate, M)
    >>> fig, axs = plt.subplots(M, sharex=True, num=1, figsize=(10,8))
    >>> for m in range(M):
    >>>      axs[m].bar(arrival_times[m], np.ones_like(arrival_times[m]))

    """
    
    arrival_times = [ [] for _ in range(M) ]
    lambda_rate_global = lambda_rate*(t1-t0) 
    n =  np.random.poisson(lambda_rate_global, size=M)
    for m in range(M):
        arrival_times[m] = np.sort(t0 + (t1-t0)*np.random.rand(n[m]))
 
    return arrival_times



def simulate_inhomogeneous_poisson(t0, t1, 
                                   lambda_rate_fn, integrated_lambda_rate_fn, 
                                   M):
    """ Simulation of homogeneours Poisson process in [t0, t1]
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    t1 : float
        Final time for the simulation
    lambda_rate_fn: callable
        Function of t that retruns the instantaneous rate
    integrated_lambda_rate_fn: callable
       Function of `s` and `t` that returns the integrated rate in [s,t]
    M: int
        Number of count sequences in simulation
            
    Returns
    -------
    times: list of M lists 
        Simulation consisting of M sequences (lists) of arrival times.
 
    Example 1
    ---------

    >>> from scipy.integrate import quad
    >>> from scipy.stats import expon
    >>> import matplotlib.pyplot as plt
    >>> import stochastic_plots as stoch
    >>> import arrival_processs_simulation as arrival

    >>> t0 = 0.0
    >>> t1 = 2000.0
    >>> M = 3 
    >>> lambda_rate = 0.5
    >>> beta_scale = 1.0/lambda_rate
    >>> def lambda_rate_fn(t): return lambda_rate
    >>> def integrated_lambda_rate_fn(s,t): return (lambda_rate * (t-s))

    >>> arrival_times = arrival.simulate_inhomogeneous_poisson(t0, t1, 
                                               lambda_rate_fn, 
                                               integrated_lambda_rate_fn, 
                                               M)
    >>> fig, axs = plt.subplots(M, sharex=True, num=1, figsize=(10,8))
    >>> for m in range(M):
            axs[m].bar(arrival_times[m], np.ones_like(arrival_times[m]))
    >>> 
    >>> interarrival_times = np.diff(arrival_times[0])
    >>> def pdf(x): return expon.pdf(x, loc = 0.0, scale = beta_scale)

    >>> stoch.plot_pdf(interarrival_times, pdf, fig_num=2)
    >>> _ = plt.xlabel('t')
    >>> _ = plt.ylabel('pdf(t)')

    Example 2
    ---------

    >>> from scipy.integrate import quad
    >>> from scipy.stats import expon
    >>> import matplotlib.pyplot as plt
    >>> import stochastic_plots as stoch
    >>> import arrival_process_simulation as arrival

    >>> t0 = 0.0
    >>> t1 = 2000.0
    >>> M = 3 
    >>> def lambda_rate_fn(t): return 1.0001 + np.sin(0.01*t)
    >>> def integrated_lambda_rate_fn(s,t): return quad(lambda_rate_fn, s, t)[0]

    >>> arrival_times = arrival.simulate_inhomogeneous_poisson(t0, t1, 
                                               lambda_rate_fn, 
                                               integrated_lambda_rate_fn, 
                                               M)
    >>> fig, axs = plt.subplots(M+2, sharex=True, num=1, figsize=(10,8))
    >>> for m in range(M):
            axs[m].bar(arrival_times[m], np.ones_like(arrival_times[m]))

    >>> n_plot = 1000
    >>> t_plot = np.linspace(t0,t1,n_plot)
    >>> _ = axs[M].hist(arrival_times[0], bins=50, density=True)
    >>> _ = axs[M+1].plot(t_plot, lambda_rate_fn(t_plot))
    
    """
    
    arrival_times = [ [] for _ in range(M) ]
    
    for m in np.arange(M):
        arrival_time = t0 
        
        while True:
            target = np.random.exponential(1)
            def f_target(t): 
                return integrated_lambda_rate_fn(arrival_time,t) - target
                       
            arrival_time = newton(f_target, x0=arrival_time, 
                                  fprime=lambda_rate_fn)     
            
            if (arrival_time > t1):
                break               
            
            arrival_times[m].append(arrival_time)

    return arrival_times



def simulate_inhomogeneous_poisson_2(t0, t1, 
                                   lambda_rate_fn,
                                   M):
    """ Simulation of homogeneours Poisson process in [t0, t1]
    
    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    t1 : float
        Final time for the simulation
    lambda_rate_fn: callable
        Function of t that retruns the instantaneous rate
    integrated_lambda_rate_fn: callable
       Function of `s` and `t` that returns the integrated rate in [s,t]
    M: int
        Number of count sequences in simulation
            
    Returns
    -------
    times: list of M lists 
        Simulation consisting of M sequences (lists) of arrival times.
 
    Example 1
    ---------

    >>> from scipy.integrate import quad
    >>> from scipy.stats import expon
    >>> import matplotlib.pyplot as plt
    >>> import stochastic_plots as stoch
    >>> import arrival_process_simulation as arrival

    >>> t0 = 0.0
    >>> t1 = 2000.0
    >>> M = 3 
    >>> lambda_rate = 0.5
    >>> beta_scale = 1.0/lambda_rate
    >>> def lambda_rate_fn(t): return lambda_rate
    >>> def integrated_lambda_rate_fn(s,t): return (lambda_rate * (t-s))

    >>> arrival_times = arrival.simulate_inhomogeneous_poisson(t0, t1, 
                                               lambda_rate_fn, 
                                               integrated_lambda_rate_fn, 
                                               M)
    >>> fig, axs = plt.subplots(M, sharex=True, num=1, figsize=(10,8))
    >>> for m in range(M):
            axs[m].bar(arrival_times[m], np.ones_like(arrival_times[m]))
    >>> 
    >>> interarrival_times = np.diff(arrival_times[0])
    >>> def pdf(x): return expon.pdf(x, loc = 0.0, scale = beta_scale)

    >>> stoch.plot_pdf(interarrival_times, pdf, fig_num=2)
    >>> _ = plt.xlabel('t')
    >>> _ = plt.ylabel('pdf(t)')

    Example 2
    ---------

    >>> from scipy.integrate import quad
    >>> from scipy.stats import expon
    >>> import matplotlib.pyplot as plt
    >>> import stochastic_plots as stoch
    >>> import arrival_process_simulation as arrival

    >>> t0 = 0.0
    >>> t1 = 2000.0
    >>> M = 3 
    >>> def lambda_rate_fn(t): return 1.0001 + np.sin(0.01*t)
    >>> def integrated_lambda_rate_fn(s,t): return quad(lambda_rate_fn, s, t)[0]

    >>> arrival_times = arrival.simulate_inhomogeneous_poisson(t0, t1, 
                                               lambda_rate_fn, 
                                               integrated_lambda_rate_fn, 
                                               M)
    >>> fig, axs = plt.subplots(M+2, sharex=True, num=1, figsize=(10,8))
    >>> for m in range(M):
            axs[m].bar(arrival_times[m], np.ones_like(arrival_times[m]))

    >>> n_plot = 1000
    >>> t_plot = np.linspace(t0,t1,n_plot)
    >>> _ = axs[M].hist(arrival_times[0], bins=50, density=True)
    >>> _ = axs[M+1].plot(t_plot, lambda_rate_fn(t_plot))
    
    """
    def rejection_sampling(lambda_rate_fn, t0,t1, N):
        n_max = 10000
        t = np.linspace(t0, t1, n_max)
        maximum_lambda_rate = np.max(lambda_rate_fn(t))
        
        arrival_times = np.zeros(N)
        n = 0
        while n < N:
            arrival_time = t0 + (t1-t0)*np.random.rand()
            threshold = lambda_rate_fn(arrival_time)
            if  (maximum_lambda_rate*np.random.rand() < threshold):
                arrival_times[n] = arrival_time
                n = n + 1
        return arrival_times        
                
     
    lambda_rate_global = quad(lambda_rate_fn,t0,t1)[0]
    n =  np.random.poisson(lambda_rate_global, size=M)
    
    arrival_times = [ [] for _ in range(M) ]

    for m in range(M):
        arrival_times[m] = np.sort(rejection_sampling(lambda_rate_fn, t0,t1, n[m]))
 
    return arrival_times