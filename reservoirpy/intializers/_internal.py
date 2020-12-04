""":mod: `reservoirpy.initilializers`, initialization tools for weight matrices.
Created on 11 juil. 2012 @author: Xavier HINAUT, xavier.hinaut #/at/# inria.fr"""
from typing import Tuple, Union

import scipy
import numpy as np

from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import eigs

from ._base import RandomSparse
from ._base import Initializer


class FastSpectralScaling(Initializer):

    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(FastSpectralScaling, self).__init__(seed)

        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.sparsity_type = sparsity_type

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        return self._fast_spectral_initialization(shape)

    def _fast_spectral_initialization(self, shape):
        if self.spectral_radius is None or self.connectivity == 0.:
            a = 1
        else:
            a = -6 * self.spectral_radius \
                / (np.sqrt(12) * np.sqrt(self.connectivity*shape[0]))

        if self.connectivity < 1:
            return sparse.random(*shape, connectivity=self.connectivity,
                                 format=self.sparsity_type,
                                 data_rvs=lambda s: self._rs.uniform(a, -a, size=s))
        else:
            return np.random.uniform(a, -a, size=shape)


class SpectralScaling(RandomSparse):

    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 distribution: str = "normal",
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 **kwargs):
        super(SpectralScaling, self).__init__(self, connectivity, distribution,
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
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 loc: float = 0.,
                 scale: float = 1.,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(NormalSpectralScaling, self).__init__(connectivity,
                                                    spectral_radius,
                                                    distribution="normal",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    loc=loc,
                                                    scale=scale)


class UniformSpectralScaling(SpectralScaling):

    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 low: float = -1.,
                 high: float = 1.,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(UniformSpectralScaling).__init__(connectivity,
                                               spectral_radius,
                                               distribution="uniform",
                                               sparsity_type=sparsity_type,
                                               seed=seed,
                                               high=high,
                                               low=low)


class BinarySpectralScaling(SpectralScaling):

    def __init__(self,
                 connectivity: float = 0.1,
                 spectral_radius: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None):
        super(BinarySpectralScaling, self).__init__(connectivity,
                                                    spectral_radius,
                                                    distribution="randint",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    high=2,
                                                    low=0)


def is_probability(proba):
    return 1. - proba >= 0. and proba >= 0.


def spectral_radius(matrix):
    if sparse.issparse(matrix):
        return max(abs(eigs(matrix, k=1, which='LM', return_eigenvectors=False)))
    return max(abs(linalg.eig(matrix)[0]))
