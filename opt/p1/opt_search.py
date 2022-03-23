# -*- coding: utf-8 -*-
"""

@author: <maria.barrosoh@estudiante.uam.es>
@author: <gloria.valle@estudiante.uam.es>
"""
import numpy as np
from typing import Callable, Tuple, List


def dichotomic_search(func: Callable[[float], float],
                      lower_bound: float, 
                      upper_bound: float, 
                      uncertainty_length: float, 
                      epsilon: float, 
                      max_iter: int = 100
)-> Tuple[float, float, int]:
    """
    Computes dicotomic search
    
    Arguments
    ---------
    func :
        Function to minimize
    lower_bound :
        Lower bound initial interval
    upper_bound :
        Upper bound initial interval
    uncertainty_length :
        Maximum length interval with optimal value
    epsilon:
        Half length uncertainty intervals
    max_iter :
        Maximum number of iterations
    
    Returns
    -------
    lower_bound : float
        Lower bound final interval
    upper_bound : float
        Upper bound final interval
    it:
        Final number of iterations computed
    
    """

    converged = False

    for it in range(max_iter):
        
        # Check stop condition
        if upper_bound - lower_bound < uncertainty_length:
            converged = True
            break

        # Define new bounds
        mid_point = 0.5 * (upper_bound + lower_bound)
        lambda_k = mid_point - epsilon
        mu_k = mid_point + epsilon

        # Update uncertainty interval
        if func(lambda_k) < func(mu_k):
            upper_bound = mu_k
        else:
            lower_bound = lambda_k

    if converged == False:
        print(f"Dichotomic search has divergenced.")

    return lower_bound, upper_bound, it


def golden_search(func: Callable[[float], float],
                  lower_bound: float, 
                  upper_bound: float, 
                  uncertainty_length: float, 
                  max_iter: int = 100
)-> Tuple[float, float, int]:
    
    """
    Computes Golden search method.
    
    Arguments
    ---------
    func :
        Function to minimize
    lower_bound :
        Lower bound initial interval
    upper_bound :
        Upper bound initial interval
    uncertainty_length :
        Maximum length interval with optimal value
    max_iter :
        Maximum number of iterations
    
    Returns
    -------
    lower_bound :
        Lower bound final interval
    upper_bound :
        Upper bound final interval
    it:
        Final number of iterations computed
    
    """

    converged = False
    golden_alpha = 0.618

    def update_lambda(lower_bound, upper_bound, golden_alpha):
        return lower_bound + (1 - golden_alpha) * (upper_bound - lower_bound)

    def update_mu(lower_bound, upper_bound, golden_alpha):
        return lower_bound + golden_alpha * (upper_bound - lower_bound)

    # Initial bounds
    lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)
    mu_k = update_mu(lower_bound, upper_bound, golden_alpha)

    for it in range(max_iter):
        # Check stop condition
        if upper_bound - lower_bound < uncertainty_length:
            converged = True
            break

        # Update bounds and uncertainty interval 
        if func(lambda_k) > func(mu_k):
            lower_bound = lambda_k
            lambda_k = mu_k
            mu_k = update_mu(lower_bound, upper_bound, golden_alpha)
        else:
            upper_bound = mu_k
            mu_k = lambda_k
            lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)

    if converged == False:
        print(f"Golden search search has divergenced.")

    return lower_bound, upper_bound, it


def HJ_search(func: Callable[[float], float],
              x0: np.ndarray, 
              lambd: float = 1.0, 
              alpha: float = 1.0, 
              beta: float = 0.5, 
              epsilon: float = 1e-3, 
              max_iter: int =100
)-> Tuple[np.ndarray, int]:
    """
    Computes Hooke-Jeeves method.
    
    Arguments
    ---------
    func :
        Function to minimize
    x0 :
        Initial point for the solution
    lambd :
        Initial detection search step
    alfa :
        alfa >= 1
        Acceleration factor
    beta:
        0 < beta < 1
        Reduction rate
    epsilon:
        Tolerance stop condition, permissible error   
    max_iter :
        Maximum number of iterations
    
    Returns
    -------
    xk :
        Optimal solution found
    it:
        Final number of iterations computed
        
    REF: 
        https://programmerclick.com/article/68471751254/
    
    """
    
    # Save initial points
    converged = False
    xk = x0.copy()
    yk = xk.copy()
    n = len(yk)
    k = 1

    for it in range(max_iter):
        # Check stop condition
        if lambd < epsilon:
            converged = True
            break
        
        # for each dimensions
        for i in range(n):
            d = np.zeros(n)
            d[i] = 1
            
            # Update optimal directions 
            if func(yk + lambd * d) < func(yk):
                yk = yk + lambd * d
            elif func(yk - lambd * d) < func(yk):
                yk = yk - lambd * d
        
        # Update points 
        if func(yk) < func(xk):
            xk = yk
            yk = yk + alpha * (yk - xk)
        else:
            lambd, yk = lambd * beta, xk

    if converged == False:
        print(f"Hooke and Jeeves search has divergenced.")

    return xk, it


def newton_search(grad: Callable[[np.ndarray], np.ndarray], 
                  hessian:  Callable[[np.ndarray], np.ndarray],
                  x0: np.ndarray,
                  lr: float = 1.0, 
                  epsilon: float = 1e-5, 
                  max_iter:int = 100
)-> Tuple[np.ndarray, int]:
    
    """
    Computes Newton search method.
    
    Arguments
    ---------
    grad :
        Gradient of the function to minimize
    hessian : 
        Hessian of the function to minimize
    x0 :
        Initial point for the solution
    lr :
        Learning rate
    epsilon :
        Tolerance stop condition, permissible error 
    max_iter :
        Maximum number of iterations
    
    Returns
    -------
    xk :
        Optimal solution found
    it:
        Final number of iterations computed
    """

    # Save initial values
    xk = x0.copy()
    converged = False

    for it in range(max_iter):
        # Update hessian matrix
        H = hessian(xk)

        # Compute H inverse
        H_inv = np.linalg.solve(H, np.eye(H.shape[0]))

        # Update next point
        xk_current = xk - lr * grad(xk) @ H_inv

        # Check stop condition
        if np.linalg.norm(xk_current - xk) < epsilon:
            converged = True
            break

        # Update next values
        xk = xk_current

    if not converged:
        print("Newton search has divergenced.")

    return xk, it


def DFP_search(grad: Callable[[np.ndarray], np.ndarray], 
               x0: np.ndarray,
               lr: float = 1.0, 
               epsilon: float = 1e-5, 
               max_iter: int = 100
)-> Tuple[np.ndarray, int]:
    
    """
    Computes Davidon-Fletcher-Powell method
    
    Arguments
    ---------
    grad :
        Gradient of the function to minimize
    x0 :
        Initial point for the solution
    lr :
        Learning rate
    epsilon :
        Tolerance stop condition, permissible error 
    max_iter :
        Maximum number of iterations
    
    Returns
    -------
    xk :
        Optimal solution found
    it:
        Final number of iterations computed
    """

    # Save initial values
    xk = x0.copy()
    Dk = np.eye(len(xk))
    converged = False

    for it in range(max_iter):
        # Compute directions
        dj = -grad(xk).T @ Dk

        # Check stop condition
        if np.linalg.norm(grad(xk)) < epsilon:
            converged = True
            break

        # Update values
        p = lr * dj
        xk_current = xk + lr * dj
        q = grad(xk_current) - grad(xk)
        
        # Compute inverse Hessian matrix
        Dk = (
            Dk
            + np.outer(p, p) / np.inner(q, p)
            - (Dk @ np.outer(q, q) @ Dk.T) / (q.T @ Dk @ q)
        )
        
        # Update next point
        xk = xk_current

    if not converged:
        print("DFP search has divergenced.")

    return xk, it


def _alpha_linesearch_secant(xk, grad, d, epsilon=1e-3, max_iter=100):
    """Computes LineSearch using secant method
    http://home.cc.umanitoba.ca/~lovetrij/cECE7670/2005/Assignments/Line%20Search.pdf"""

    # Save alpha
    alpha_cur = 0
    alpha = 0.5

    # Compute derivative gradient
    der_x0 = grad(xk).T @ d
    der = der_x0

    for it in range(max_iter):
        # Compute new step
        alpha_old = alpha_cur
        alpha_cur = alpha
        der_old = der
        der = grad(xk + alpha_cur * d).T @ d

        # Check stop condition
        if der < epsilon:
            break

        # Save next alpha
        alpha = (der * alpha_old - der_old * alpha_cur) / (der - der_old)

        # Check final stop condition
        if np.abs(der) < epsilon * np.abs(der_x0):
            break

    return alpha


def fletcher_reeves(grad, x0, epsilon=1e-3, max_iter=100, tolerance=1e-8):
    """Computes conjugate gradient descent using Fletcher-Reeves method"""

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
        xk = xn

    return xk, it
