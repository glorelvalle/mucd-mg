from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


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
    K: np.ndarray
    ) -> np.ndarray:
    """
    Centering of the kernel

    Parameters
    ----------
    K:
        Kernel matrix

    Returns
    -------
    Gram matrix of the centered kernel

    Example
    -------
    >>> 
    """

    # Get number of elements
    N = K.shape[0]

    # Save ones vector
    ones_vector = np.ones((N, N))

    # Compute given formula for centered kernel matrix form
    return K - 1/N * (K@ones_vector) - 1/N * (ones_vector@K) + 1/(N**2) * (ones_vector@K@ones_vector)


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

    # NOTE <YOUR CODE HERE>.

    K = kernel(X, X)

    KC = centered_kernel(K)

    lambda_eigenvals, alpha_eigenvecs = np.linalg.eigh(KC)

    lambda_eigenvals, alpha_eigenvecs = lambda_eigenvals[::-1], alpha_eigenvecs[:, ::-1]

    print(lambda_eigenvals, alpha_eigenvecs)

    X_test_hat = X_test


    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
