
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.gaussian_process.kernels import RBF # RBF kernel as default method in Nystroem
from sklearn.base import BaseEstimator, TransformerMixin # Mixin and Base classes for all transformers and estimators in scikit-learn.



class RandomFeaturesSampler(ABC, BaseEstimator, TransformerMixin):
    """ Base class for random feature samplers. """

    def __init__(self, n_random_features: int = 100) -> None:
        self.n_random_features = n_random_features
        self.w = None

    def _initialize_w(self, n_features: int) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, n_features: int) -> np.ndarray:
        """Initialize w's for the random features.
        This should be implemented for each kernel."""
        pass

    def fit_transform(
        self,
        X: np.ndarray,
        X_prime: Optional[np.ndarray] = None # Added to ensure compatibility with sklearn
    ) -> np.ndarray:
        """Initialize  w's (fit) & compute random features (transform)."""
        n_features = np.shape(X)[1]
        self.fit(n_features)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute the random features.
        Assumes that the vector of w's has been initialized

        Parameters
        ----------
        X:
            Data matrix of shape (n_instances, n_features).

        Returns
        -------
        random_features:
            Array of shape (n_instances, n_random_features).
        """
        if (self.w is None):
            raise ValueError('Use fit_transform to initialize w.')

        n_instances, n_features = np.shape(X)

        if (np.shape(self.w)[1] != n_features):
            raise ValueError('Different # of features for X and w.')

        random_features = np.empty((n_instances, self.n_random_features))
        random_features[:, ::2] = np.cos(X @ self.w.T)
        random_features[:, 1::2] = np.sin(X @ self.w.T)

        norm_factor = np.sqrt(self.n_random_features // 2)
        random_features = random_features / norm_factor

        return random_features
    

class RandomFeaturesSamplerRBF(RandomFeaturesSampler):
    """ Random Fourier Features for the RBF kernel. """

    def __init__(self, sigma: float = 1.0, n_random_features: int = 100) -> None:
        self.sigma = sigma # sigma_kernel modified to sigma to ensure compatibility with pipeline grid search
        super().__init__(n_random_features) # Initializes the number of random features of the inherited class

    def fit(self, n_features: int) -> np.ndarray:
        """Initialize the w's for the random features."""
        w_mean = np.zeros(n_features)
        w_cov_matrix = (1.0/self.sigma**2) * np.identity(n_features)

        rng = np.random.default_rng()
        self.w = rng.multivariate_normal(
            w_mean,
            w_cov_matrix,
            self.n_random_features // 2,
        )



class RandomFeaturesSamplerMatern(RandomFeaturesSampler):
    """Random Fourier Features for the Matern kernel."""

    def __init__(self, scale: float = 1.0, nu: float = 1.5, n_random_features: int = 100) -> None:
        """The Fourier transform of the Matérn kernel is a
        Student's t distribution with twice the degrees of freedom.
        Ref. Chapter 4 of
        Carl Edward Rasmussen and Christopher K. I. Williams. 2005.
        Gaussian Processes for Machine Learning
        (Adaptive Computation and Machine Learning). The MIT Press. There is probably a mistake with the scale factor.
        """

        self.nu = nu # Modified Matèrn nu parameter to ensure compatibility with pipeline grid search
        self.scale = scale # Modified length_scale parameter to scale to ensure compatibility with pipeline grid search

        super().__init__(n_random_features) # Initializes the number of random features of the inherited class
        self.w = None

    def fit(self, n_features: int) -> np.ndarray:
        """Compute w's for random Matérn features."""
        # Scale of the Fourier tranform of the kernel
        w_mean = np.zeros(n_features)
        w_cov_matrix = (1.0/self.scale**2) * np.identity(n_features) 
        
        self.w = random_multivariate_student_t(
            w_mean,
            w_cov_matrix,
            2.0 * self.nu, 
            self.n_random_features // 2,
        )


def random_multivariate_student_t(
    mean: np.ndarray,
    cov_matrix: np.ndarray,
    degrees_of_freedom: float,
    n_samples: int,
) -> np.ndarray:
    """Generate samples from a multivariate Student's t.
    https://en.wikipedia.org/wiki/Multivariate_t-distribution#Definition
    """

    # Dimensions of multivariate Student's t distribution.
    D = len(mean)

    rng = np.random.default_rng()
    x = rng.chisquare(degrees_of_freedom, n_samples) / degrees_of_freedom

    Z = rng.multivariate_normal(
        np.zeros(D),
        cov_matrix,
        n_samples,
    )

    X = mean + Z / np.sqrt(x)[:, np.newaxis]
    return X

class NystroemFeaturesSampler(BaseEstimator, TransformerMixin):
    """Sample Nystroem features. """

    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = RBF(),
        n_random_features: int = 100
    ) -> None:
        self.kernel = kernel # Modified _kernel parameter to kernel to ensure compatibility with pipeline grid search
        self.component_indices_ = None
        self._X_reduced = None
        self._reduced_kernel_matrix = None
        self.n_random_features = n_random_features # Added n_random_features in the class constructor to ensure classes compatibility

    def fit(self, X: np.ndarray, 
            y: Optional[np.ndarray] = None # Added to ensure compatibility with sklearn
           ) -> np.ndarray:
        """Precompute auxiliary quantities for Nystroem features.
        """
        
        n_instances = len(X)
        # Check random features dimenstions and data samples dimenstions to avoid kernel problems computing distances
        if self.n_random_features > n_instances:
            n_random_features = n_instances
            warnings.warn("n_random_features > n_samples, n_random_features was set to n_instances.")
        else:
            n_random_features = self.n_random_features

        
        # Sample subset of training instances.
        rng = np.random.default_rng()
        self.component_indices_ = rng.choice(
            range(n_instances),
            size=n_random_features,
            replace=False,
        )

        self._X_reduced = X[self.component_indices_, :]

        # Compute reduced kernel matrix.
        self._reduced_kernel_matrix = self.kernel(
            self._X_reduced,
            self._X_reduced
        )

        self._reduced_kernel_matrix = (
            self._reduced_kernel_matrix + self._reduced_kernel_matrix.T
        ) / 2.0  # enforce symmetry of kernel matrix

        # Compute auxiliary quantities.
        self._sqrtm_pinv_reduced_kernel_matrix = sp.linalg.sqrtm(
            np.linalg.pinv(
                self._reduced_kernel_matrix,
                rcond=1.0e-6,
                hermitian=True
            )
        )

        # Check that complex part is negligible and eliminate it
        if np.iscomplexobj(self._sqrtm_pinv_reduced_kernel_matrix):
            threshold_imaginary_part = 1.0e-6
            max_imaginary_part = np.max(
                np.abs(np.imag(self._sqrtm_pinv_reduced_kernel_matrix))
            )
            if max_imaginary_part > threshold_imaginary_part:
                warnings.warn(
                    'Maximum imaginary part is {}'.format(max_imaginary_part)
                )

            self._sqrtm_pinv_reduced_kernel_matrix = np.real(
                self._sqrtm_pinv_reduced_kernel_matrix
            )
            
        return self # Return the class to ensure compatibility with sklearn

    # Esta funcion no se llama nunca pero en caso de que se llame, calcularía correctamente la matriz de Kernel aproximada
    def approximate_kernel_matrix(
        self,
        X: np.ndarray,
        X_prime: Optional[np.ndarray] = None 
    ) -> np.ndarray:
        """Approximate the kernel matrix using Nystroem features."""

        # NOTE <YOUR CODE HERE>.
        X_features = self.fit_transform(X)
        return X_features @ X_features.T
    
    def fit_transform(
        self,
        X: np.ndarray,
        X_prime: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Nyström features."""
        self.fit(X, self.n_random_features)
        X_nystroem = self.transform(X)
        return X_nystroem
    
    def transform(self, X_prime: np.ndarray) -> np.ndarray:
        """Compute Nystroem features with precomputed quantities."""
        
        # NOTE <YOUR CODE HERE>.
        J = self.kernel(X_prime, self._X_reduced)
        return J@self._sqrtm_pinv_reduced_kernel_matrix


def demo_kernel_approximation_features(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler: Union[RandomFeaturesSampler, NystroemFeaturesSampler],
    n_features: np.array,
) -> None:
    """Kernel approximation using random Fourier features (RFF)."""
    n_plots = len(n_features) + 1
    fig, axes = plt.subplots(1, n_plots)
    fig.set_size_inches(15, 4)
    font = {'fontname': 'arial', 'fontsize': 18}

    kernel_matrix = kernel(X, X)
    axes[0].imshow(kernel_matrix, cmap=plt.cm.Blues)
    axes[0].set_title('Exact kernel', **font)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    for n, ax in zip(n_features, axes[1:]):
        print('# of features = ', n)
        
        features_sampler.set_params(n_random_features=n) # Modified number of random features of each class
        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T

        ax.imshow(kernel_matrix_approx, cmap=plt.cm.Blues)

        err_approx = kernel_matrix - kernel_matrix_approx
        err_mean = np.mean(np.abs(err_approx))
        err_max = np.max(np.abs(err_approx))
        
        ax.set_xlabel('err (mean) = {:.4f} \n err (max) = {:.4f}'.format(
            err_mean,
            err_max
        ), **font)

        ax.set_title('{} features'.format(n), **font)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    plt.show()
    
def demo_kernel_approximation_error_evolution(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler: Union[RandomFeaturesSampler, NystroemFeaturesSampler],
    N: int # Upper bound on the number of simulations,
) -> None:
    """Error evolution of Kernel approximation."""

    kernel_matrix = kernel(X, X)
    
    err_means = []
    features_range = range(2, N+1, 2)
    for n in features_range:
        features_sampler.set_params(n_random_features=n)
        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T
        err_means.append(np.mean(np.abs(kernel_matrix - kernel_matrix_approx)))
    
    plt.plot(features_range, err_means, label = 'empirical error')
    if isinstance(features_sampler, NystroemFeaturesSampler):
        plt.plot(features_range, 1/np.power(features_range, 1.25), label = 'theoretical error Nystrom')
    else:
        plt.plot(features_range, 1/np.sqrt(features_range), label = 'theoretical error Monte Carlo')

    plt.legend()
    plt.title("Evolution of mean approximation error")
    plt.xlabel("n features sampled")
    plt.ylabel("mean absolute error")
    
    

if __name__ == '__main__':

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, svm
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.metrics.pairwise import rbf_kernel

    import kernel_approximation as ka

    # A not so simple 2 D problem
    X, Y = datasets.make_moons(n_samples=100, noise=0.3, random_state=0)

    # Compute grid of points for plotting the decision regions
    grid_x, grid_y = np.meshgrid(
        np.linspace(-3, 3, 50),
        np.linspace(-3, 3, 50),
    )

    grid_X = np.c_[grid_x.ravel(), grid_y.ravel()]

    gamma = 0.5
    # Kernel matrix
    def kernel(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)

    n_nystroem_features = 20

    nystroem_sampler = NystroemFeaturesSampler(kernel)
    nystroem_features = nystroem_sampler.fit_transform(n_nystroem_features, X)
    nystroem_features_grid = nystroem_sampler.transform(grid_X)

    clf = svm.SVC(kernel='linear')
    # clf = svm.NuSVC(gamma='auto')
    clf.fit(nystroem_features, Y)

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.metrics.pairwise import rbf_kernel

    # 3-D data
    n_instances = 1000
    X, t = datasets.make_s_curve(n_instances, noise=0.1)
    X = X[np.argsort(t)]

    # 2-D data
    # X, y = datasets.make_moons(n_samples=400, noise=.05)
    # X = X[np.argsort(y)]

    # Reshape if necessary
    if (X.ndim == 1):
        X = X[:, np.newaxis]

    # Kernel parameters
    sigma = 1.0
    gamma = 1.0 / (2.0 * sigma**2)

    n_nystroem_features = [10, 100, 1000]

    # Kernel function
    def kernel(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)


    nystroem_features = NystroemFeaturesSampler(kernel)

    demo_kernel_approximation_features(
        X,
        kernel,
        nystroem_features,
        n_nystroem_features,
    )
