from typing import Union, Callable, Tuple

import numpy as np

from numpy.random import RandomState

from .initializers import initializer


class Reservoir(object):

    def __init__(self,
                 shape: Union[Tuple[Tuple[int], int], Tuple[int]],
                 leak: int,
                 spectral_radius: float,
                 input_scaling: Union[float, Tuple[float]],
                 connectivity: float,
                 input_connectivity: Union[float, Tuple[float]],
                 activation: Union[str, Callable],
                 w_init: Union[str, Callable],
                 win_init: Union[str, Callable],
                 noise: Union[float, Callable],
                 input_noise: Union[float, Callable],
                 noise_dist: Union[str, Callable],
                 Win,
                 W,
                 sparsity_type: str,
                 seed: Union[int, RandomState]):
        self.shape = shape
        self.leak = leak
        self.spectral_radius = spectral_radius,
        self.input_scaling = input_scaling
        self.connectivity = connectivity
        self.input_connectivity = input_connectivity
        self.activation = activation
        self.w_init = w_init
        self.win_init = win_init
        self.noise = noise
        self.input_noise = input_noise
        self.noise_dist = noise_dist
        self.sparsity_type = sparsity_type,

        self.reset_seed(seed)

        self.Win = Win
        self.W = W

    def __call__(self,
                 inputs,
                 feedback=None,
                 init_state=None,
                 noise_seed=None):
        ...

    def _set_w_initializer(self, arg):
        if callable(arg):
            self.w_init_func = arg
        elif isinstance(arg, str):
            self.w_init_func = initializer(arg,
                                           connectivity=self.connectivity,
                                           spectral_radius=self.spectral_radius,
                                           seed=self._rs)

    def _set_win_initializer(self, arg):
        if callable(arg):
            self.win_init_func = arg
        elif isinstance(arg, str):
            self.win_init_func = initializer(arg,
                                             connectivity=self.input_connectivity,
                                             scaling=self.input_scaling,
                                             seed=self._rs)

    def _build_internal(self):
        self.W = self.win_init_func((self.shape[1], self.shape[1]))

    def _build_input(self):
        # Win shape : (neurons, inputs)
        self.Win = self.win_init_func((self.shape[1], self.shape[0]))

    def _build(self):
        ...

    def reset_seed(self, seed: Union[int, RandomState]):
        if isinstance(seed, RandomState):
            self._rs = seed
        else:
            self._rs = RandomState(seed)
        self.seed = seed
