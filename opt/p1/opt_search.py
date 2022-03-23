# -*- coding: utf-8 -*-
"""
Tarea 1. 2. Actividades:
 - Codificar el método de búsqueda Dicotómica.
 - Codificar el método de búsqueda de la Sección Áurea.

@author: <maria.barrosoh@estudiante.uam.es>
"""
import numpy as np


def dichotomic_search(
    func, lower_bound, upper_bound, uncertainty_length, epsilon, max_iter=100
):

    converged = False

    for it in range(max_iter):

        if upper_bound - lower_bound < uncertainty_length:
            converged = True
            break

        mid_point = 0.5 * (upper_bound + lower_bound)
        lambda_k = mid_point - epsilon
        mu_k = mid_point + epsilon

        if func(lambda_k) < func(mu_k):
            upper_bound = mu_k
        else:
            lower_bound = lambda_k

    if converged == False:
        print(f"Dichotomic search has divergenced.")

    return lower_bound, upper_bound, it


def golden_search(func, lower_bound, upper_bound, uncertainty_length, max_iter=100):

    converged = False
    golden_alpha = 0.618

    def update_lambda(lower_bound, upper_bound, golden_alpha):
        return lower_bound + (1 - golden_alpha) * (upper_bound - lower_bound)

    def update_mu(lower_bound, upper_bound, golden_alpha):
        return lower_bound + golden_alpha * (upper_bound - lower_bound)

    lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)
    mu_k = update_mu(lower_bound, upper_bound, golden_alpha)

    for it in range(max_iter):

        if upper_bound - lower_bound < uncertainty_length:
            converged = True
            break

        if func(lambda_k) > func(mu_k):
            lower_bound = lambda_k
            lambda_k = mu_k
            mu_k = update_mu(lower_bound, upper_bound, golden_alpha)
        else:
            upper_bound = mu_k
            mu_k = lambda_k
            lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)

    if converged == False:
        print(f"Dichotomic search has divergenced.")

    return lower_bound, upper_bound, it


# https://programmerclick.com/article/68471751254/
def HJ_search(func, x0, lambd=1.0, alpha=1.0, beta=0.5, epsilon=1e-3, max_iter=100):
    """
    x0 : punto inicial
    lambd : paso de búsqueda de detección inicial
    alfa (alfa> = 1) : factor de aceleración
    beta (0 <beta <1) : tasa de reducción
    epsilon  (epsilon> 0) : error permisible
    """
    converged = False
    xk = x0.copy()
    yk = xk.copy()
    n = len(yk)
    k = 1

    for it in range(max_iter):
        if lambd < epsilon:
            converged = True
            break

        for i in range(n):
            d = np.zeros(n)
            d[i] = 1

            if func(yk + lambd * d) < func(yk):
                yk = yk + lambd * d
            elif func(yk - lambd * d) < func(yk):
                yk = yk - lambd * d

        if func(yk) < func(xk):
            xk = yk
            yk = yk + alpha * (yk - xk)
        else:
            lambd, yk = lambd * beta, xk

    if converged == False:
        print(f"Hooke and Jeeves search has divergenced.")

    return xk, it


def newton_search(grad, hessian, x0, lr=1.0, epsilon=1e-5, max_iter=100):

    xk = x0.copy()
    converged = False

    for it in range(max_iter):
        H = hessian(xk)
        H_inv = np.linalg.solve(H, np.eye(H.shape[0]))

        xk_current = xk - lr * grad(xk) @ H_inv

        if np.linalg.norm(xk_current - xk) < epsilon:
            converged = True
            break

        xk = xk_current

    if not converged:
        print("Newton search has divergenced.")

    return xk, it


def DFP_search(func, grad, x0, lr=1.0, epsilon=1e-5, max_iter=100):

    xk = x0.copy()
    n = len(xk)
    Dk = np.eye(n)
    converged = False

    for it in range(max_iter):
        dj = -grad(xk).T @ Dk

        if np.linalg.norm(grad(xk)) < epsilon:
            converged = True
            break

        p = lr * dj

        xk_current = xk + lr * dj
        q = grad(xk_current) - grad(xk)

        Dk = (
            Dk
            + np.outer(p, p) / np.inner(q, p)
            - (Dk @ np.outer(q, q) @ Dk.T) / (q.T @ Dk @ q)
        )

        xk = xk_current

    if not converged:
        print("DFP search has divergenced.")

    return xk, it


def _alpha_linesearch_secant(xk, grad, d, epsilon=1e-3, max_iter=100):
    alpha_cur = 0
    alpha = 0.5

    der_x0 = grad(xk).T @ d
    der = der_x0

    for it in range(max_iter):
        alpha_old = alpha_cur
        alpha_cur = alpha
        der_old = der
        der = grad(xk + alpha_cur * d).T @ d

        if der < epsilon:
            break

        alpha = (der * alpha_old - der_old * alpha_cur) / (der - der_old)

        if np.abs(der) < epsilon * np.abs(der_x0):
            break

    return alpha


def fletcher_reeves(grad, x0, epsilon=1e-3, max_iter=100, tolerance=1e-8):

    # Initial conditions
    xk = x0.copy()
    converged = False
    alpha = 0

    for it in range(max_iter):
        # Set gradient function
        g = grad(xk)

        # Stop condition
        if np.linalg.norm(g) < epsilon:
            converged = True
            break

        # Save directional vector first it
        if it == 0:
            d = -g
            g_old = g

        # Coefficient for calculating conjugate directional vector
        beta = (g.T @ g) / (g_old.T @ g_old)

        # New directional vector
        d = -g + beta * d

        # Step size
        alpha = _alpha_linesearch_secant(xk, grad, d)

        xn = xk + alpha * d

        if np.linalg.norm(xn - xk) < tolerance * np.linalg.norm(xk):
            print("Tolerance in xk is reached in %d iterations." % (it))
            break

        # Save actual values
        g_old = g
        xk = xk + alpha * d

    return xk, it
