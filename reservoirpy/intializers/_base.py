""":module: reservoirpy.intializers._base provides base
utility for initializer definition.
"""

from typing import Tuple, Union
from abc import ABC, abstractmethod
from functools import partial

import scipy
import numpy as np

from scipy import sparse
from numpy.random import RandomState


class Initializer(ABC):
    """Base class for weights initializers. All initializers should
    inherit from this class.

    All initializers should implement their own ``__call__`` method::
        def __call__(self, shape):
            # returns a matrix with the specifiyed shape
            # this matrix should be either of type numpy.ndarray
            # or scipy.sparse

    Parameters:
    -----------
    seed: int
        Random state generator seed.

    Attributes:
    -----------
    seed: int
        Random state generator seed.
    _rs: :class: ~numpy.random.RandomState
        Numpy RandomState instance. Used for reproducibility.

    Example
    -------
    Here is an example of an initializer building a sparse matrix
    with discrete values between 0 and 1::

        class BinaryInitializer(Initializer):

            def __init__(self, density, seed):
                super(BinaryInitializer, self).__init__(seed)

                # a random generator seed is required as
                # argument by the base Initializer class.
                # the random state is then
                # available in self._rs

            def __call__(self, shape):
                distribution = lambda s: self._rs.uniform(0, 2, size=s)
                return scipy.sparse.random(*shape,
                                            density=self.density,
                                            data_rvs=distribution)
    """
    def __init__(self, seed):
        self.reset_seed(seed)

    @abstractmethod
    def __call__(self, shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        """Produce a matrix of specifiyed shape.
        Abstract method defined by all the subclasses
        of :class: `reservoirpy.initializers.Initializer` .

        Parameters
        ----------
        shape : tuple (dim1, dim2)
            Shape of the matrix to build.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated matrix.
            Can be either in a sparse or a dense format.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def reset_seed(self, seed: Union[int, RandomState] = None):
        """Produce a new numpy.random.RandomState object based
        on the given seed. This RandomState generator will then
        be used to compute the random matrices.

        Parameters:
        -----------
        seed: int or RandomState, optional
            If set, will be used to randomly generate the matrices.
        """
        if type(seed) is RandomState:
            self.seed = None
            self._rs = seed
        else:
            self.seed = seed
            self._rs = RandomState(seed)


class RandomSparse(Initializer):
    """Random sparse matrices generator object.

    Parameters
    ----------
    connectivity : float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    distribution: str, defaults to "normal"
        A numpy.random.RandomState distribution function name.
        Usual distributions are "normal", "uniform", "standard_normal",
        or "randint" with high=2 and low=0, to draw samples from a discrete
        uniform distribution between 0 and 1.
    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState

    **kwargs: optional
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 distribution: str = "normal",
                 sparsity_type: str = "csr",
                 seed: Union[int, RandomState] = None,
                 **kwargs):
        super(RandomSparse, self).__init__(seed)

        self.connectivity = connectivity
        self.sparsity_type = sparsity_type
        self.distribution = distribution

        self._set_distribution(distribution, **kwargs)

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Union[np.ndarray, scipy.sparse]:
        return self._normal_spectral_initialization(shape)

    def _set_distribution(self, distribution, **kwargs):
        data_rvs = getattr(self._rs, distribution)
        self._partial_data_rvs = partial(data_rvs, **kwargs)
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def _initialize(self, shape):

        if self.connectivity < 1:
            return sparse.random(*shape, density=self.connectivity,
                                 format=self.sparsity_type,
                                 data_rvs=self._partial_data_rvs)
        else:
            return self._partial_data_rvs(size=shape)

    @property
    def seed(self):
        return self.seed

    @seed.setter
    def seed(self, value: Union[int, RandomState]):
        self.reset_seed(value)
        self._set_distribution(self.distribution,
                               **self._partial_data_rvs.keywords)
