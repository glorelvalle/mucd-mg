import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import arrival_process_simulation as arrival
from scipy import stats, integrate, special
from BM_simulators import *
from stochastic_plots import *

def move_animation(frame, ax, B, upper, lower, res1, res2):
    ax[0].cla()
    ax[1].cla()
    ax[0].set_xlim(t0, t0 + T)
    ax[0].set_ylim(lower, upper)
    ax[1].set_xlim(lower, upper)
    ax[1].set_ylim(0, 1)
    plt.tight_layout()
    ax[0].plot(t[:frame+1], B[:, :frame+1].T)
    ax[0].plot(t[frame:], B[:, frame:].T, alpha=0.25)
    ax[0].axvline(t[frame], color='k', linestyle='--')
    ax[1].hist([b[frame] for b in B], bins=res1, density=True, alpha=0.8, color = 'cadetblue', label=f'Empirical')

    if frame > 0:
        ax[1].plot(res2, stats.norm.pdf(res2, loc=B0 + mu*(t[frame]-t0), scale=sigma*np.sqrt(t[frame]-t0)), color='k', linestyle='--', label='Theoretical')
    else:
        ax[1].axvline(B0, color='k', linestyle='--')
        ax[1].axhline(0, color='k', linestyle='--')
    ax[1].legend()


sns.set_style('whitegrid')

mu, sigma = 1.5, 1

t0 = 0          # initial time
T = 1           # total length
B0 = 2          # initial point

M = 1000        # number of trajectories
N = 100         # number of steps

# Simulation
t, B = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)

# Animation parameters
lower, upper = np.min(B), np.max(B)
fig, ax = plt.subplots(1, 2, figsize=(15,8))
fig.suptitle('Brownian Motion')
res1, res2 = np.linspace(lower, upper, 15), np.linspace(lower, upper, 100)
# Brown motion
gif = animation.FuncAnimation(fig, move_animation, N, fargs=(ax, B, upper, lower, res1, res2))
plt.show()