""":mod: `reservoirpy.initilializers._internal`
Provides base tools for internal reservoir weights initialization.
"""
from typing import Union, Tuple

import scipy
import numpy as np

from numpy.random import RandomState
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csc import csc_matrix

from ._base import RandomSparse


class FastSpectralScaling(RandomSparse):
    """Fast Spectral Initialization (FSI) technique for
    reservoir internal weights.

    Quickly performs spectral radius scaling and
    returns matrices in a sparse format.

    The weigths follows:

    .. math::
        W \distas U(-a, a)

    where:

    .. math::
        \rhoW = \max |eigenvalues(W)|
        a = -6 \frac{spectralradius}{\sqrt{12}\sqrt{connectivity \cdot Nbneurons}}


    Usage:
    ------
        >>> fsi = FastSpectralScaling(spectral_radius=0.9)
        >>> fsi((5, 5)  # generate a 5x5 weight matrix

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    spectral_radius: float, optional
        Maximum eigenvalue of the initialized matrix.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Parameters:
    -----------
    connectivity : float, defaults to 0.1

    spectral_radius: float, optional

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    distribution: {"uniform"}
        FSI requires an uniform distribution of weights value.

    seed: int or RandomState instance
    """
    @property
    def seed(self):
        """
        int: a random state generator seed.
        """
        return self._seed

    @seed.setter
    def seed(self, value: Union[int, RandomState]):
        #  reset only the RandomState in the case of FSI
        #  distribution function is already reset at each call
        self.reset_seed(value)

    @property
    def distribution(self):
        """
        str: a RandomState random statistical
        distribution generator function name.
        """
        return self._distribution

    @distribution.setter
    def distribution(self, value, **kwargs):
        #  during FSI, distribution must always be uniform.
        pass

    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, RandomState] = None):
        # uniform distribution between -1 and 1 by default. this will
        # change at each call.
        super(FastSpectralScaling, self).__init__(connectivity, 'uniform',
                                                  sparsity_type, seed,
                                                  low=-1, high=1)
        self.spectral_radius = spectral_radius

    def __call__(self,
                 shape: Tuple[int, int],
                 ) -> Union[np.ndarray, csr_matrix, csc_matrix, coo_matrix]:
        """Produce a random sparse matrix of representing the
        weights of a specifyed number of neuronal units.

        Parameters
        ----------
        shape : tuple of int
            Shape of weights matrix.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated weights.
            Can be either in a sparse or a dense format,
            depending on the connectivity parameter set in
            the initializer.
        """
        # adapt FSI coef to the current reservoir shape
        a = self._fsi_coeff(shape[0])

        # reset the distribution function accordingly
        self._set_distribution(self._distribution, high=a, low=-a)

        return super(FastSpectralScaling, self).__call__(shape)

    def _fsi_coeff(self, units):
        """Compute FSI coefficient ``a`` (see class documentation).
        """
        if self.spectral_radius is None or self.connectivity == 0.:
            return 1
        else:
            return -6 * self.spectral_radius \
                / (np.sqrt(12) * np.sqrt(self.connectivity*units))


class SpectralScaling(RandomSparse):
    """Weight initialization with spectral radius scaling.

    The weigths follows any specifyed distribution, and are then
    rescaled:

    .. math::
        W := W \frac{spectralradius}{\rho_W}

    where:

    .. math::
        \rhoW = \max |eigenvalues(W)|


    Usage:
    ------
        >>> sr_scaling = SpectralScaling(distribution="normal",
        ...                              loc=0, scale=1,
        ...                              spectral_radius=0.9)
        >>> sr_scaling((5, 5)  # generate a (5, 5) weight matrix

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    spectral_radius: float, optional
        Maximum eigenvalue of the initialized matrix.

    distribution: str, defaults to "normal"
        A numpy.random.RandomState distribution function name.
        Usual distributions are "normal", "uniform", "standard_normal",
        or "choice" with ``a=[-1, 1]``, to randomly draw -1 or 1.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    **kwargs: optional
        Keywords arguments to pass to the numpy.random.RandomState
        distribution function.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    spectral_radius: float, optional

    distribution: str, defaults to "normal"

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 distribution: str = "normal",
                 sparsity_type: str = "csr",
                 seed: Union[int, RandomState] = None,
                 **kwargs):
        super(SpectralScaling, self).__init__(connectivity, distribution,
                                              sparsity_type, seed, **kwargs)

        self.spectral_radius = spectral_radius

    def __call__(self,
                 shape: Tuple[int, int],
                 ) -> Union[np.ndarray, csr_matrix, csc_matrix, coo_matrix]:
        """Produce a random sparse matrix of representing the
        weights of a specifyed number of neuronal units.

        Parameters
        ----------
        shape : tuple of int
            Shape of weight matrix.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated weights.
            Can be either in a sparse or a dense format,
            depending on the connectivity parameter set in
            the initializer.
        """
        matrix = super().__call__(shape)
        return self.spectral_scaling(matrix)

    def spectral_scaling(self,
                         matrix: Union[csc_matrix, coo_matrix,
                                       csr_matrix, np.ndarray]
                         ) -> Union[csr_matrix,
                                    np.ndarray,
                                    csc_matrix,
                                    coo_matrix]:
        """Rescale a matrix to a specific spectral radius.

        Parameters
        ----------
        matrix : Scipy sparse matrix or Numpy array

        Returns
        -------
        Scipy sparse matrix or Numpy array
            Scaled matrix.
        """
        if self.spectral_radius is not None:
            rhoW = spectral_radius(matrix)
            matrix *= spectral_radius / rhoW

        return matrix


class NormalSpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and normal distribution
    of weights value.

    Usage:
    ------
        Same as :class: `SpectralScaling`

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    spectral_radius: float, optional
        Maximum eigenvalue of the initialized matrix.

    loc: float, defaults to 0
        Mean of the distribution

    scale: float, default to 1
        Standard deviation of the distribution

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    spectral_radius: float, optional

    loc: float, default to 0

    scale: float, default to 1

    distribution: {"normal"}

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 loc: float = 0.,
                 scale: float = 1.,
                 sparsity_type: str = "csr",
                 seed: Union[int, RandomState] = None):
        super(NormalSpectralScaling, self).__init__(connectivity,
                                                    spectral_radius,
                                                    distribution="normal",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    loc=loc,
                                                    scale=scale)


class UniformSpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and uniform distribution
    of weights value.

    Usage:
    ------
        Same as :class: `SpectralScaling`

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    spectral_radius: float, optional
        Maximum eigenvalue of the initialized matrix.

    high, low: float, defaults to (-1, 1)
        Boundaries of the uniform distribution of weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    spectral_radius: float, optional

    high: float, defaults to 1

    low: float, defaults to -1

    distribution: {"uniform"}

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 low: float = -1.,
                 high: float = 1.,
                 sparsity_type: str = "csr",
                 seed: Union[int, RandomState] = None):
        super(UniformSpectralScaling).__init__(connectivity,
                                               spectral_radius,
                                               distribution="uniform",
                                               sparsity_type=sparsity_type,
                                               seed=seed,
                                               high=high,
                                               low=low)


class BinarySpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and random discrete
    distribution of weights over two values.

    Usage:
    ------
        Same as :class: `SpectralScaling`.

        >>> bin_spectral = BinarySpectralScaling(spectral_radius=0.9,
        ...                                       val1=-1, val2=1)
        >>> bin_spectral(5)

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    spectral_radius: float, optional
        Maximum eigenvalue of the initialized matrix.

    val1, val2: float, defaults to (-1, 1)
        Authorized values for the weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    spectral_radius: float, optional

    val1: float, defaults to 1

    val2: float, defaults to -1

    distribution: {"choice"}
        ``numpy.random.RandomState.choice([val1, val2])`` is used
        to generate the weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 sparsity_type: str = "csr",
                 val1: float = -1,
                 val2: float = 1,
                 seed: Union[int, np.random.RandomState] = None):
        super(BinarySpectralScaling, self).__init__(connectivity,
                                                    spectral_radius,
                                                    distribution="choice",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    a=[val1, val2])


def is_probability(proba):
    return 1. - proba >= 0. and proba >= 0.


def spectral_radius(matrix):
    if sparse.issparse(matrix):
        return max(abs(eigs(matrix, k=1, which='LM', return_eigenvectors=False)))
    return max(abs(linalg.eig(matrix)[0]))
