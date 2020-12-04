from typing import Union

import numpy as np

from ._base import RandomSparse


class NormalScaling(RandomSparse):

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


class BinaryScaling(RandomSparse):

    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = None,
                 sparsity_type: str = "csr",
                 seed: Union[int, np.random.RandomState] = None,
                 ):
        super(BinaryScaling, self).__init__(self,
                                            connectivity,
                                            distribution="choice",
                                            sparsity_type=sparsity_type,
                                            seed=seed,
                                            a=[-1, 1])
        self.scaling = scaling
