""":mod: `reservoirpy.intializers._base` provides base
utility for initializer definition.
"""
from typing import Tuple, Union
from abc import ABC

import numpy as np

from scipy import sparse
from scipy.sparse.csr import csr_matrix
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
    seed: int or RandomState instance, optional
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
    def __init__(self, seed: Union[int, RandomState] = None):
        self.reset_seed(seed)

    def __call__(self, shape: Tuple[int, int]
                 ) -> Union[np.ndarray, csr_matrix]:
        raise NotImplementedError

    @property
    def seed(self):
        return self._seed

    def reset_seed(self, seed: Union[int, RandomState] = None):
        """Produce a new numpy.random.RandomState object based
        on the given seed. This RandomState generator will then
        be used to compute the random matrices.

        Parameters:
        -----------
        seed: int or RandomState instance, optional
            If set, will be used to randomly generate the matrices.
        """
        if isinstance(seed, RandomState):
            self._rs = seed
        else:
            self._rs = RandomState(seed)
        self.seed = seed


class RandomSparse(Initializer):
    """Random sparse matrices generator object.

    Usage:
    ------
        >>> sparse_initializer = RandomSparse(connectivity=0.2,
        ...                                   distribution="normal",
        ...                                   loc=0, scale=1)
        >>> sparse_initializer((5, 5))  # generate a (5, 5) matrix

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    distribution: str, defaults to "normal"
        A numpy.random.RandomState distribution function name.
        Usual distributions are "normal", "uniform", "standard_normal",
        or "choice" with ``a=[-1, 1]``, to randomly draw -1 or 1.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    **kwargs: optional
        Keywords arguments to pass to the numpy.random.RandomState
        distribution function.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    distribution: str, defaults to "normal"

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance

    Otheir attributes set in ``kwargs`` parameter and used to describe
    the chosen distribution, like the ``loc`` and ``scale`` parameters of
    the 'normal' distribution.
    """
    @property
    def seed(self):
        """
        int: a random state generator seed.
        """
        return self._seed

    @seed.setter
    def seed(self, value: Union[int, RandomState]):
        #  reset the RandomState and the distribution function
        #  when resetting the seed
        self.reset_seed(value)
        self._set_distribution(self._distribution,
                               self._rvs_kwargs)

    @property
    def distribution(self):
        """
        str: a RandomState random statistical
        distribution generator function name.
        """
        return self._distribution

    @distribution.setter
    def distribution(self, value: str, **kwargs):
        #  reset the distribution
        self._distribution = value
        self._set_distribution(self._distribution, **kwargs)

    def __init__(self,
                 connectivity: float = 0.1,
                 distribution: str = "normal",
                 sparsity_type: str = "csr",
                 seed: Union[int, RandomState] = None,
                 **kwargs):
        super(RandomSparse, self).__init__(seed=seed)

        self.connectivity = connectivity
        self.sparsity_type = sparsity_type

        self._distribution = distribution

        #  partial function to draw random samples
        #  initialized with kwargs
        self._rvs_kwargs = None
        self._rvs = None

        self._set_distribution(distribution, **kwargs)

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Union[np.ndarray, csr_matrix]:
        """Produce a random sparse matrix of specifiyed shape.

        Parameters
        ----------
        shape : tuple (dim1, dim2)
            Shape of the matrix to build.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated matrix.
            Can be either in a sparse or a dense format,
            depending on the connectivity parameter set in
            the initializer.
        """
        return self._initialize_dist(shape)

    def _set_distribution(self, distribution, **kwargs):
        """Produce a partial function for a parametrized
        distribution function.
        """
        data_rvs = getattr(self._rs, distribution)

        # partially initialize the distribution function
        # all random values will be sampled from _rvs
        def _rvs(size=1):
            return data_rvs(size=size, **kwargs)

        self._rvs_kwargs = kwargs
        self._rvs = _rvs

        # store the distribution parameters as attributes
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def _initialize_dist(self, shape):
        """Returns a matrix initialized following
        the specifiyed distribution.

        Parameters
        ----------
        shape : (dim1, dim2)

        Returns
        -------
        np.ndarray or scipy.sparse
        """
        if self.connectivity < 1:  # sparse
            return sparse.random(*shape, density=self.connectivity,
                                 random_state=self._rs,
                                 format=self.sparsity_type,
                                 data_rvs=self._rvs)
        else:  # dense
            return self._rvs(size=shape)
