from typing import Optional, Callable, Tuple

import numpy as np
from pyparsing import alphanums
from scipy.spatial import distance
from sklearn.utils.extmath import svd_flip


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkovski', p=1.0)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)

def centered_kernel(
    K: np.ndarray,
    K_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """
    Centering of the kernel

    Parameters
    ----------
    K:
        Kernel matrix
    
    K_test:

        Kernel test matrix (Optional, for phase 4 Kernel PCA)

    Returns
    -------
    Gram matrix of the centered kernel

    Example
    -------
    >>> 
    """

    # Get number of elements (N)
    N = K.shape[0]

    # Get number of elements (L)
    L = K_test.shape[0] if K_test is not None else N

    # Save ones vector
    ones_vector = np.ones((N, N))

    # Compute given formula for centered kernel matrix form
    if L == N:
        K_res = K - 1/N * (K @ ones_vector) - 1/N * (ones_vector @ K) + 1/(N**2) * (ones_vector @ K @ ones_vector)
    else:
        ones_vector_p = np.ones((L, N))
        K_res = K_test - 1/N * (K_test @ ones_vector) - 1/N*(ones_vector_p @ K) + 1/(N**2) * (ones_vector_p @ K @ ones_vector)
    
    return K_res


def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    A:
        output variance
    ls:
        kernel lengthscale

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """

    # Kernel Gram matrix
    K = kernel(X, X)

    # Centered kernel gram matrix
    K_tilde = centered_kernel(K)

    # Compute eigenvalues and eigenvectors
    lambda_eigenvals, alpha_eigenvecs = np.linalg.eigh(K_tilde)

    # Find the first d non-zero eigenvalues and the corresponding eigenvectors
    lambda_eigenvals[lambda_eigenvals < 1.0e-9] = 0.
    lambda_eigenvals, alpha_eigenvecs = lambda_eigenvals[::-1], alpha_eigenvecs[:, ::-1]

    # Save non-zero eigenvalues
    lambda_nonzero = np.flatnonzero(lambda_eigenvals)

    # Kernel test gram matrix
    K_test = kernel(X_test, X)

    # Compute centered kernel test gram matrix
    K_test_tilde = centered_kernel(K, K_test)

    # Sign correction of eigenvectors
    V_t = np.zeros_like(alpha_eigenvecs).T
    alpha_eigenvecs, _ = svd_flip(alpha_eigenvecs, V_t)
    
    # Normalization of eigenvectors
    alpha_eigenvecs[:, lambda_nonzero] /= np.sqrt(lambda_eigenvals[lambda_nonzero])

    # Output the projections of the test data onto the first d components
    X_test_hat = K_test_tilde @ alpha_eigenvecs[:, lambda_nonzero]

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
