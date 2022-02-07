# -*- coding: utf-8 -*-
"""
Plotting utilities for the trajectories of stochastic processes

@author: <alberto.suarez@uam.es>

"""

# Load packages
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(t, X, 
                      max_trajectories=20, 
                      fig_num=1, fig_size=(8, 4), 
                      fontsize=14, mean_color='k'):
    """ Plots a sample of trajectories and their mean """

    M, _ = np.shape(X)
    
    # Plot trajectories 
    M = np.min((M, max_trajectories))
    fig, ax = plt.subplots(1, 1, num=fig_num)
    ax.plot(t, X[:M, :].T, linewidth=1)
    ax.set_xlabel('t', fontsize=fontsize)
    ax.set_ylabel('X(t)', fontsize=fontsize)
    ax.set_title('Simulation', fontsize=fontsize)
  
    # Plot mean
    ax.plot(t, np.mean(X, axis=0), linewidth=3, color=mean_color)
  
    return fig, ax
  
    
def plot_pdf(X, pdf,
             max_bins=50,
             fig_num=1, fig_size=(4,4), fontsize=14):
    """ Compare pdf with the normalized histogram.
        The normalized histogram is an empirical estimate of the pdf.
    """

    # Plot histogram

    fig, ax = plt.subplots(1, 1, num=fig_num)

    n_samples = len(X)

    n_bins = np.min((np.int(np.sqrt(n_samples)), 
                    max_bins))
       
    ax.hist(X, bins=n_bins, density=True)
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('pdf(x)', fontsize=fontsize)
    
    # Compare with exact distribution
    n_plot = 1000
    x_plot = np.linspace(np.min(X), np.max(X), n_plot)
    y_plot = pdf(x_plot)
    ax.plot(x_plot, y_plot, linewidth=2, color='r')
    
    return fig, ax


