""":mod: `reservoirpy.initilializers._input_feedback`
Provides base tools for input and feedback weights initialization.
"""
from typing import Union

import numpy as np

from ._base import RandomSparse


class NormalScaling(RandomSparse):
    """Class for input and feedback
    weights initialization following a normal
    distribution. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient change the standard deviation
    of the normal distribution from which the weights are
    sampled. The mean of this distribution remains 0.

    For input weights initialization, shape of the returned matrix
    should always be (resservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Usage:
    ------
        >>> norm_scaling = NormalScaling(connectivity=0.2,
        ...                              scaling=0.5)
        >>> norm_scaling((10, 3))

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    scaling: float, optional
        Scaling coefficient to apply on the weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    scaling: float, optional

    loc: {0}

    distribution: {"normal"}

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(NormalScaling, self).__init__(self,
                                            connectivity,
                                            distribution="normal",
                                            sparsity_type=sparsity_type,
                                            seed=seed,
                                            loc=0,
                                            scale=scaling)


class UniformScaling(RandomSparse):
    """Class for input and feedback
    weights initialization following an uniform
    distribution. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient change the boundaries of the
    uniform distribution from which the weights are
    sampled.

    For input weights initialization, shape of the returned matrix
    should always be (resservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Usage:
    ------
        >>> uni_scaling = UniformScaling(connectivity=0.2,
        ...                              scaling=0.5)
        >>> uni_scaling((10, 3))

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    scaling: float, optional
        Scaling coefficient to apply on the weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    scaling: float, optional

    high, low: {scaling, -scaling}

    distribution: {"uniform"}

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(UniformScaling, self).__init__(self,
                                             connectivity,
                                             distribution="uniform",
                                             sparsity_type=sparsity_type,
                                             seed=seed,
                                             low=-scaling, high=scaling)


class BivaluedScaling(RandomSparse):
    """Class for input and feedback
    weights initialization with only two
    values, -1 and 1. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient is applyed over
    the dicrete values chosen for the weights.

    For input weights initialization, shape of the returned matrix
    should always be (resservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Usage:
    ------
        >>> bin_scaling = BivaluedScaling(connectivity=0.2,
        ...                               scaling=0.5)
        >>> bin_scaling((10, 3))

    Parameters:
    -----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.

    scaling: float, optional
        Scaling coefficient to apply on the weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
        scipy.sparse format to use.

    seed: int
        Random state generator seed.

    Attributes:
    -----------
    connectivity : float, defaults to 0.1

    scaling: float, optional

    distribution: {"choice"}
    ``numpy.random.RandomState.choice([val1, val2])`` is used
        to generate the weights.

    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"

    seed: int or RandomState instance
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(BivaluedScaling, self).__init__(self,
                                              connectivity,
                                              distribution="choice",
                                              sparsity_type=sparsity_type,
                                              seed=seed,
                                              a=[-1*scaling, 1*scaling])
        self.scaling = scaling
