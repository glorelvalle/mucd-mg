import numpy as np
import matplotlib.pyplot as plt
import arrival_process_simulation as arrival
from scipy import stats, integrate, special
from BM_simulators import *
from stochastic_plots import *

def poisson_pmf_est_theo(lambda_rate, M, T, t0, t1):
    """ Simulates Poisson distribution and computes theoretical PMF in [t0,t1]

        P_{t;n} = (1/n!) * lambda^n * t^n * e^(-\lambda*t)

    Parameters
    ----------
    lambda_rate : int
        Lambda poisson parameter
    M : int
        Number of steps for the simulation
    T : float
        Length of the simulation
    t0 : float
        Initial time for the simulation
    t1 : float
        Time for the simulation

    Returns
    -------
    pmf_est : numpy.ndarray
        PMF Poisson estimated value
    pmf_theo : numpy.ndarray
        PMF Poisson theoretical value

    Example
    -------
    >>> import arrival_process_simulation as arrival
    >>> from scipy import stats, integrate, special
    >>> lambda_rate, T, M = 10, 41, 50000
    >>> t0, t1 = 0., 2.
    >>> pmf_est, pmf_theo = poisson_pmf_est_theo(lambda_rate, M, T, t0, t1)

    """
    # Simulated distribution
    arrival_times = arrival.simulate_poisson(t0, t1, lambda_rate, M)
    N = np.array([len(a) for a in arrival_times])
    pmf_est = np.bincount(N, minlength=T)/M

    # Theoretical distribucion
    X = np.arange(len(pmf_est))
    pmf_theo = (lambda_rate*t1)**X * np.exp(-lambda_rate*t1) / special.factorial(X)
    
    return pmf_est, pmf_theo

def plot_poisson_pmf(lambda_rate, M, T, t0, t1):
    """ Plots Poisson distribution and theoretical PMF in [t0,t1]

    Parameters
    ----------
    lambda_rate : int
        Lambda poisson parameter
    M : int
        Number of steps for the simulation
    T : float
        Length of the simulation
    t0 : float
        Initial time for the simulation
    t1 : float
        Time for the simulation
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import arrival_process_simulation as arrival
    >>> from scipy import stats, integrate, special
    >>> lambda_rate, T, M = 10, 41, 50000
    >>> t0, t1 = 0., 2.
    >>> plot_poisson_pmf(lambda_rate, M, T, t0, t1)
    
    """
    # Get PMF estimation and theoretical value
    pmf_est, pmf_theo = poisson_pmf_est_theo(lambda_rate, M, T, t0, t1)
    X = np.arange(len(pmf_est))
    
    # Plot
    width = 0.35
    plt.figure(figsize=(15, 8))
    plt.title('Poisson Process Simulation vs Probability Density')
    plt.bar(X-width, pmf_theo, color='royalblue', width=width, label='Theoretical', align='center')
    plt.bar(X, pmf_est, color='plum', width=width, label='Simulation', align='center')
    plt.xlabel(f'$n$ events')
    plt.ylabel(f'$P(N(2)=n)$')
    plt.legend()
    plt.show()

def erlang(t, n, lambda_rate):
    """ Computes Erlang PDF

        f_{S_n}(t)= lambda^n * t^(n-1) * e^(-lambda*t) / (n-1)!
    
    Parameters
    ----------
    t : float
        Given t value for Erlang
    n : int
        Ghape factor for Erlang PMF
    lambda_rate : float
        Lambda parameter for Erlang PMF
    
    Returns
    -------
    numpy.ndarray
        Erlang PDF theoretical value

    """
    return lambda_rate**n * t**(n-1)* np.exp(-lambda_rate*t) / special.factorial(n-1)

def plot_poisson_erlang(lambda_rate, M, N, t0, t1):
    """ Plots Poisson distribution and theoretical PMF in [t0,t1]

    Parameters
    ----------
    lambda_rate : int
        Lambda poisson parameter
    M : int
        Number of steps for the simulation
    N : numpy.array
        Given events for the simulation
    T : float
        Length of the simulation
    t0 : float
        Initial time for the simulation
    t1 : float
        Time for the simulation
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import arrival_process_simulation as arrival
    >>> from scipy import stats, integrate, special
    >>> lambda_rate, T, M, N = 10, 5, 50000, N = np.array([1, 2, 5, 10])
    >>> t0, t1 = 0., 7.
    >>> plot_poisson_erlang(lambda_rate, M, N, T, t0, t1)
    
    """

    # Poisson simulation
    arrival_times = arrival.simulate_poisson(t0, t1, lambda_rate, M)
    arrival_n = [[a[n-1] for a in arrival_times] for n in N]

    # Theoretical density (Erlang)
    ts = np.linspace(0, t1, 100)
    f_sn = [erlang(ts, n, lambda_rate) for n in N]

    # Plots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    axs = axs.flatten()
    for i in range(4):
        axs[i].hist(arrival_n[i], color='cornflowerblue', density=True, bins=50, label='Simulation', alpha=0.5)
        axs[i].plot(ts, f_sn[i], color='indigo', label = 'Erlang')
        axs[i].set_xlabel('$s_{' + str(N[i]) +'}$')
        axs[i].set_ylabel('pdf$(s_{' + str(N[i]) + '})$')
        axs[i].set_title(f'Density of {N[i]} arrival')
        axs[i].legend()

def poisson_simulate_n1_n2(lambda_rate1, lambda_rate2, M, t0, t1):
    """ Poisson simulation for N1 and N2 in [t0,t1]

    Parameters
    ----------
    lambda_rate1 : float
        Lambda rate for Poisson (N2)
    lambda_rate2 : float
        Lambda rate for Poisson (N2)
    M : int
        Number of steps of the simulation
    t0 : float
        Initial time for the simulation
    t1 : float
        Time for the simulation

    Returns
    -------
    N : np.array
        Addition of N1 and N2
    N1 : np.array
        First random variable simulation
    N2 : np.array
        Second random variable simulation
    arrival_rimes1: list
        Arrival times (N1)
    arrival_times2: list
        Arrival times (N2)

    Example
    -------
    >>> import arrival_process_simulation as arrival
    >>> lambda_rate1, lambda_rate2, T, M = 0.02, 0.03, 5, 50000
    >>> t0, t1 = 0., 7.
    >>> poisson_simulate_n1_n2(lambda_rate1, lambda_rate2, M, t0, t1)
    """

    # Simulation arrival times N1 y N2
    arrival_times1 = arrival.simulate_poisson(t0, t1, lambda_rate1, M)
    arrival_times2 = arrival.simulate_poisson(t0, t1, lambda_rate2, M)

    # Get N1, N2 and N = N1 + N2
    N1 = np.array([len(a) for a in arrival_times1])
    N2 = np.array([len(a) for a in arrival_times2])
    N = N1 + N2

    return N, N1, N2, arrival_times1, arrival_times2

def simulate_wiener(t0, T, M, N):
    """
    Simulation of standard Wiener process

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : int
        Length of the simulation
    M : int
        Number of trajectories
    N : int
        Number of steps for the simulation

    Returns
    -------
    t : numpy.ndarray
        Regular grid of discretization times [0,1]
    X(t) : numpy.ndarray
        Wiener Process

    Examples
    --------
    >>> import numpy as np
    >>> t, X = simulate_wiener(t0, T, M, N)
        
    """
    t = np.linspace(t0, t0+T, N+1)          # grid
    Z = np.random.randn(M).reshape(M, 1)    # gaussian white noise
    sqrt_t = np.sqrt(t).reshape(1, N+1) 
    return t, np.dot(Z, sqrt_t)

def plot_simulate_wiener(t, X):
    """ Plots a simulation for standard Wiener Process
    
    Parameters
    ----------
    t : numpy.ndarray
        Regular grid of discretization times [0,1]
    X(t) : numpy.ndarray
        Wiener Process
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plot_simulate_wiener(t, X)

    """
    fig, ax = plt.subplots(1, figsize=(15, 8))
    fig.suptitle('Process $X(t)$  for $t_0 = 0$, $T = 1$, $M = 100$, $N = 100$')
    ax.plot(t, X.T)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$x(t)$')
    plt.show()
    
def gamma(r, s, t0, B, T):
    """
    Returns [0,1] covariance matrix for Wiener processes

    Parameters
    ----------
    r : float
        Initial time (grid)
    s : float
        Final time
    t0 : float
        Initial time for simulation
    B0 : float
        Initial level of the process
    T : float
        Length of the simulation 

    Returns
    -------
    numpy.ndarray (covariance matrix in [0,1])

    Examples
    --------
    >>> import numpy as np
    >>> from BM_simulators import *
    >>> t, B = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> M, N = 1000, 500
    >>> empirical = [gamma(i, time, 0, B, T) for i in t]

    """

    assert(r <= (T+t0) and r >= t0)
    assert(s <= (T+t0) and s >= t0)
    n = len(B[0]) - 1           # number of time intervals
    i_r = int((r-t0)*n/T)       # idx r
    i_s = int((s-t0)*n/T)       # idx s

    return np.cov(B[:, i_r], B[:, i_s])[0,1]

def wiener_emp_theo(time, t0, B0, T, M, N):
    """ Simulation in [0,1] of Wiener process
        Computes empirical and theoretical value

    Parameters
    ----------
    time : float
        Time simulation
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
    empirical : numpy.array
        Empirical value
    theoretical : numpy.array
        Theoretical value
    t : numpy.ndarray
        Regular grid of discretization times
    """

    # Simulation (gaussian parameters)
    mu, sigma = 0, 1
    t, B = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)

    # Compute empirical and theoretical value
    empirical = [gamma(i, time, 0, B, T) for i in t]
    theoretical = np.where(t < time, t, time)

    return empirical, theoretical, t

def plot_wiener_emp_theo(empirical, theoretical, t):
    """ Plots Empirical vs Theoretical value of Wiener process

    Parameters
    ----------
    empirical : np.array
        Empirical value
    theoretical : np.array
        Theoretical value

    Examples
    --------
    >>> import numpy as np
    >>> from BM_simulators import *
    >>> empirical, theoretical = wiener_emp_theo(time, t0, B0, T, M, N)
    >>> plot_wiener_emp_theo(empirical, theo, t)

    """
    # Plots
    plt.figure(figsize=(15, 8))
    plt.plot(t, theoretical, label="Theoretical value", color='tomato')
    plt.plot(t, empirical, label="Empirical value", color='indigo')
    plt.xlabel('t')
    plt.ylabel('$\gamma(W(t)W(0.25))$')
    plt.title('Empirical vs Theoretical Autocovariance')
    plt.show()

def plot_wiener_simul_pdf(time, example, label, length, B0, mu, sigma):
    """ Plots Simulation and PDF from given Wiener process
    
    Parameters
    ----------
    time : numpy.ndarray
        Regular grid of discretization times
    example : numpy.ndarray
        Given Wiener process
    label : str
        Wiener process name
    length : float
        Length of simulation
    B0 : float
        Initial level of the process
    mu, sigma : float
        Parameters of the process

    Examples
    --------
    >>> from BM_simulators import *
    >>> mu, sigma, B0, T = 0, 1, 0, 1.
    >>> rho = 0.5
    >>> t, W = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> _, W_ = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> A = rho*W + np.sqrt(1 - rho**2)*W_
    >>> plot_wiener_simul_pdf(t, A, f'$V_1(t)$', T, B0, mu, sigma)
    """
    
    # Simulation plot
    plot_trajectories(time, example)
    plt.ylabel(label)
    plt.show()
    # PDF plot
    pdf = lambda f: stats.norm.pdf(f, B0 + mu*length, sigma*np.sqrt(length))
    plot_pdf(example[:,-1], pdf)
    plt.ylabel(label)
    plt.title('PDF')
    plt.show()