from typing import Union, Callable, Sequence

import numpy as np

from numpy.random import RandomState
from scipy import sparse

from .initializers import initializer

Matrix = Union[np.ndarray,
               sparse.csc.csc_matrix,
               sparse.csr.csr_matrix,
               sparse.coo.coo_matrix]


def to_tuple(obj):
    if not hasattr(obj, "__len__"):
        return (obj, )
    else:
        return obj


class Reservoir(object):

    def __init__(self,
                 units: int,
                 inputs_dim: Union[int, Sequence[int]],
                 lr: float = 1.0,
                 sr: float = None,
                 iss: Union[float, Sequence[float]] = 1.0,
                 inputs_bias: Union[bool, Sequence[bool]] = True,
                 inputs_co: Union[float, Sequence[float]] = 0.1,
                 co: float = 0.1,
                 inputs_init: Union[Callable,
                                    Sequence[Callable],
                                    str,
                                    Sequence[str]
                                    ] = "binary",
                 init: Union[Callable, str] = "fsi",
                 activation: Union[Callable, str] = "tanh",
                 W: Matrix = None,
                 Win: Union[Matrix, Sequence[Matrix]] = None,
                 seed: Union[int, RandomState] = None
                 ):

        self._units = units
        self._inputs_dim = to_tuple(inputs_dim)
        self._sr = sr
        self._iss = iss
        self._inputs_bias = to_tuple(inputs_bias)
        self._inputs_co = to_tuple(inputs_co)
        self._co = co
        self._inputs_init = to_tuple(inputs_init)
        self._init = init
        self._seed = seed

        self.activation = activation
        self.lr = lr
        self.W = W
        self.Win = Win

        self.state = None
        self.inputs_nb = len(self._inputs_dim)

        self._init_func = None
        self._input_init_func = []

    def _initialize(self):
        ...

    def _check_input_dimensions(self, init_param):
        return len(init_param) == 1 or len(init_param) == self.inputs_nb

    def _check_dimensions(self):
        if not(self._check_input_dimensions(self._inputs_init)):
            raise ValueError(f"there is {self.inputs_nb} different inputs "
                             f"but only {len(self._inputs_init)} input "
                             "initializers.")
        if not(self._check_input_dimensions(self._inputs_co)):
            raise ValueError(f"there is {self.inputs_nb} different inputs "
                             f"but only {len(self._inputs_co)} input "
                             "connectivities.")

        if not(self._check_input_dimensions(self._inputs_bias)):
            raise ValueError(f"there is {self.inputs_nb} different inputs "
                             f"but only {len(self._inputs_bias)} input "
                             "biases initializers.")

    def _check_reservoir_init(self):
        if type(self._init) is str:
            self._init_func = initializer(self._init,
                                          connectivity=self._co,
                                          spectral_radius=self._sr,
                                          seed=self._seed)
        elif callable(self._init):
            self._init_func = self._init

    def _check_input_init(self):
        if not hasattr(self._input_init, "__len__"):
            self._input_init = (self._input_init, )

        for in_init in self._input_init:
            if type(in_init) is str:
                # self._input_init_func = initializer(self._input_init,
                #                                     connectivity=self._input_co,
                #                                     )
                ...

    def _initialize_reservoir(self):
        if self.W is None:
            ...

    def __call__(self, inputs, fb_from):
        pass
