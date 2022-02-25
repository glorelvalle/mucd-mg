# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 22:20:47 2020

@author: María Barroso and Gloria del Valle

This script allows the Kernel PCA animation execution.

"""
# Load packages
import sys
import seaborn as sns
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import kernel_machine_learning as KML


def plot_config(ax, gamma):
    """ """
    ax.clear()
    ax.set_xlabel(r"$x_1$: first principal component in space induced by $\phi$")
    ax.set_ylabel(r"$x_2$: second principal component")
    ax.set_title(r"Projection by KPCA, $\gamma=$" + f"{gamma:.1f}")


def move_animation(N, gammas, ax, X, X_test, reds, blues, A, L):
    """ """
    gamma = gammas[N]
    L = np.sqrt(0.5 / gamma)

    # Local Kernel definition
    kernel = lambda X, X_p: KML.rbf_kernel(X, X_p, A, L)

    plot_config(ax, gamma)

    X_kpca, _, _ = KML.kernel_pca(X, X_test, kernel)

    ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue", s=20, edgecolor="k")


def animate(X, X_test, y_test, N, gammas, A, L):
    """ """

    # Init
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # For plot results
    reds = y_test == 0
    blues = y_test == 1

    # Generate Kernel PCA animation
    return animation.FuncAnimation(
        fig,
        move_animation,
        N,
        repeat=False,
        fargs=(gammas, ax, X, X_test, reds, blues, A, L),
    )


# For reproducible results
seed = 0
np.random.seed(seed)

# Input data
X, y = datasets.make_moons(n_samples=400, noise=0.05, random_state=seed)

# Init animation parameters
A, L = 1.0, 1.0
N = 50

# Init gammas
gammas = 2 * np.logspace(-5, 5, N)

# Save animation
gif = animate(X, X, y, N, gammas, A, L)
gif.save("KPCA.gif", writer="imagemagick", fps=20)
plt.show()
