""":mod: `reservoirpy.initilializers`, initialization tools for weight matrices.
Created on 11 juil. 2012 @author: Xavier HINAUT, xavier.hinaut #/at/# inria.fr"""
import time
import warnings

from functools import partial
from typing import Tuple, Union
from abc import ABC

import scipy
import numpy as np

from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import eigs


class Initializer(ABC):
    """Base class for weights initializers. All initializers should
    inherit from this class.

    All initializers should implement their own ``__call__`` method::
        def __call__(self, shape):
            # returns a matrix with the specifiyed shape
            # this matrix should be either of type numpy.ndarray
            # or scipy.sparse

    Example
    -------
    Here is an example of an initializer building a sparse matrix
    with discrete values between 0 and 1::

        class BinaryInitializer(Initializer):

            def __init__(self, density, seed):
                self.density = density

                # reset_seed is provided to help
                # ensure reproducibility. it reseeds
                # a random state.
                self.reset_seed(seed)

                # the random state is then
                # available in self._rs

            def __call__(self, shape):
                distribution = lambda s: self._rs.uniform(0, 2, size=s)
                return scipy.sparse.random(*shape,
                                            density=self.density,
                                            data_rvs=distribution)
    """
    @abstractmethod
    def __call__(self, shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        raise NotImplementedError

    def reset_seed(self, seed: int = None):
        """Produce a new :py:class: `numpy.random.RandomState` object based
        on the given seed.

        Parameters:
        -----------
        :param seed: if None, randomize the state. Defaults to None.
        :type seed: int, optional
        """
        self.seed = seed
        self._rs = np.random.RandomState(seed)


class FastSpectralScaling(Initializer):

    def __init__(self,
                 density: float = 0.1,
                 spectral_radius: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        self.density = density
        self.spectral_radius = spectral_radius
        self.sparsity_type = sparsity_type

        self.reset_seed(seed)

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        return self._fast_spectral_initialization(shape)

    def _fast_spectral_initialization(self, shape):
        if self.spectral_radius is None or self.density == 0.:
            a = 1
        else:
            a = -6 * self.spectral_radius \
                / (np.sqrt(12) * np.sqrt(self.density*shape[0]))

        if self.density < 1:
            return sparse.random(*shape, density=self.density,
                                 format=self.sparsity_type,
                                 data_rvs=lambda s: self._rs.uniform(a, -a, size=s))
        else:
            return np.random.uniform(a, -a, size=shape)


class RandomSparse(Initializer):

    def __init__(self,
                 density: float = 0.1,
                 distribution: str = "normal",
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 **kwargs):

        self.density = density
        self.sparsity_type = sparsity_type

        self.reset_seed(seed)
        self._set_distribution(distribution, **kwargs)

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        return self._normal_spectral_initialization(shape)

    def _set_distribution(self, distribution, **kwargs):
        self.distribution = distribution
        data_rvs = getattr(self._rs, distribution)
        self._partial_data_rvs = partial(data_rvs, **kwargs)
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def _initialize(self, shape):

        if self.density < 1:
            return sparse.random(*shape, density=self.density,
                                 format=self.sparsity_type,
                                 data_rvs=self._partial_data_rvs)
        else:
            return self._partial_data_rvs(size=shape)


class SpectralScaling(RandomSparse):

    def __init__(self,
                 density: float = 0.1,
                 spectral_radius: float = None,
                 distribution: str = "normal",
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 **kwargs):
        super(SpectralScaling, self).__init__(self, density, distribution,
                                              sparsity_type, seed, **kwargs)

        self.spectral_radius = spectral_radius

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        matrix = super().__call__(shape)
        return self._spectral_scaling(matrix)

    def _spectral_scaling(self, matrix):
        if self.spectral_radius is not None:
            rhoW = spectral_radius(matrix)
            matrix *= spectral_radius / rhoW

        return matrix


class NormalSpectralScaling(SpectralScaling):

    def __init__(self,
                 density: float = 0.1,
                 spectral_radius: float = None,
                 loc: float = 0.,
                 scale: float = 1.,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(NormalSpectralScaling, self).__init__(density,
                                                    spectral_radius,
                                                    distribution="normal",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    loc=loc,
                                                    scale=scale)


class UniformSpectralScaling(SpectralScaling):

    def __init__(self,
                 density: float = 0.1,
                 spectral_radius: float = None,
                 low: float = -1.,
                 high: float = 1.,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(UniformSpectralScaling).__init__(density,
                                               spectral_radius,
                                               distribution="uniform",
                                               sparsity_type=sparsity_type,
                                               seed=seed,
                                               high=high,
                                               low=low)


class BinarySpectralScaling(SpectralScaling):

    def __init__(self,
                 density: float = 0.1,
                 spectral_radius: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(BinarySpectralScaling, self).__init__(density,
                                                    spectral_radius,
                                                    distribution="randint",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    high=2,
                                                    low=0)


class NormalScaling(RandomSparse):

    def __init__(self,
                 density: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(NormalScaling, self).__init__(self,
                                            density,
                                            distribution="normal",
                                            sparsity_type=sparsity_type,
                                            seed=seed,
                                            loc=0,
                                            scale=scaling)


class UniformScaling(RandomSparse):

    def __init__(self,
                 density: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(UniformScaling, self).__init__(self,
                                             density,
                                             distribution="uniform",
                                             sparsity_type=sparsity_type,
                                             seed=seed,
                                             low=-scaling, high=scaling)


class BinaryScaling(RandomSparse):

    def __init__(self,
                 density: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(BinaryScaling, self).__init__(self,
                                            density,
                                            distribution="uniform",
                                            sparsity_type=sparsity_type,
                                            seed=seed,
                                            low=0, high=2)
        self.scaling = scaling

    def __call__(self, shape: Tuple[int, int]):
        matrix = super(BinaryScaling, self).__call__(shape)
        return matrix *= self.scaling


def is_probability(proba):
    return 1. - proba >= 0. and proba >= 0.


def spectral_radius(matrix):
    if sparse.issparse(matrix):
        return max(abs(eigs(matrix, k=1, which='LM', return_eigenvectors=False)))
    return max(abs(linalg.eig(matrix)[0]))
