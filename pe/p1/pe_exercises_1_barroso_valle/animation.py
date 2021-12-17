# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 22:20:47 2020

@author: María Barroso and Gloria del Valle

This script allows the Brownian Motion animation execution.

"""
# Load packages
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import arrival_process_simulation as arrival
from matplotlib import rc
from scipy import stats, integrate, special
from BM_simulators import *
from stochastic_plots import *

def move_animation(frame, ax, B, upper, lower, res1, res2):
    """ Updates animation steps and cleans figure axes

    Parameters
    ----------
    frame : int
        Current animation frame
    ax : AxesSubplot
        Axes iterable
    B : array
        BM data
    upper : float
        Axis max limit
    lower : float
        Axis min limit
    res1 : array
        Bins for histogram
    res2 : array
        Points for PDF
    
    """
    ax[0].cla()
    ax[1].cla()

    ax[0].set_xlabel(f'Time $t$')
    ax[0].set_ylabel(f'$B(t)$')

    ax[1].set_xlabel(f'$B(t)$')
    ax[1].set_ylabel('Frequency')

    ax[0].set_xlim(t0, t0 + T)
    ax[0].set_ylim(lower, upper)

    ax[1].set_xlim(lower, upper)
    ax[1].set_ylim(0, 1)

    plt.tight_layout()

    ax[0].plot(t[:frame+1], B[:, :frame+1].T)
    ax[0].plot(t[frame:], B[:, frame:].T, alpha=0.25)
    ax[0].axvline(t[frame], color='k', linestyle='--')

    ax[1].set_title(f'$t$ = {t[frame]:.2}')
    ax[1].hist([b[frame] for b in B], bins=res1, density=True, alpha=0.8, color = 'cadetblue', label=f'Empirical')

    if frame > 0:
        ax[1].plot(res2, stats.norm.pdf(res2, loc=B0 + mu*(t[frame]-t0), scale=sigma*np.sqrt(t[frame]-t0)), color='k', linestyle='--', label='Theoretical')
    else:
        ax[1].axhline(0, color='k', linestyle='--')
        ax[1].axvline(B0, color='k', linestyle='--')

    ax[1].legend()


sns.set_style('whitegrid')

mu, sigma = 1.5, 1

t0 = 0          # initial time
T = 1           # total length
B0 = 2          # initial point

M = 1000        # number of trajectories
N = 50          # number of steps (try 100)

# Simulation
t, B = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)

# Animation parameters
lower, upper = np.min(B), np.max(B)
fig, ax = plt.subplots(1, 2, figsize=(15,8))
fig.suptitle(f'Brownian Motion for $\mu = {mu}, \sigma = {sigma}, M = {M}, N = {N}, t_0 = {t0}, B_0 = {B0}, T = {T}$')
res1, res2 = np.linspace(lower, upper, 30), np.linspace(lower, upper, 100)

# Brownian motion
rc('animation', html='html5')
gif = animation.FuncAnimation(fig, move_animation, N, fargs=(ax, B, upper, lower, res1, res2))
gif.save('BM_animation.gif', writer='imagemagick', fps=60)
plt.show()