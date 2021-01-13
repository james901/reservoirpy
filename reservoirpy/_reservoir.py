from typing import Union, Callable, Tuple

import numpy as np

from numpy.random import RandomState

from .initializers import initializer


class Reservoir(object):

    def __init__(self,
                 shape: Union[Tuple[Tuple[int], int], Tuple[int]],
                 leak: float,
                 spectral_radius: float,
                 bias: bool,
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

        self._build()

        self.state = None

    def __call__(self,
                 inputs,
                 feedback=None,
                 teachers=None,
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
        self._build_input()
        self._build_internal()

    def _run(self, inputs, teachers, readout, init_state, noise_seed):
        responses = np.zeros(inputs.shape[0], self.shape[1])
        for t in range(inputs.shape[0]):
            teacher = None
            if teachers is not None:
                teacher = teachers[t]
            fb = readout.feedback(teachers)
            responses[t] = self._next_state(inputs[t], fb)

    def _next_state(self,
                    single_input: np.ndarray,
                    feedback: np.ndarray = None,
                    last_state: np.ndarray = None) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t), compute the state vector x(t+1).

        Arguments:
            single_input {np.ndarray} -- Input vector u(t).

        Keyword Arguments:
            feedback {np.ndarray} -- Feedback vector if enabled. (default: {None})
            last_state {np.ndarray} -- Current state to update x(t). If None,
                                       state is initialized to 0. (default: {None})

        Raises:
            RuntimeError: feedback is enabled but no feedback vector is available.

        Returns:
            np.ndarray -- Next state x(t+1).
        """
        # first initialize the current state of the ESN
        if last_state is None:
            x = np.zeros((self.shape[1], 1),dtype=self.typefloat)
        else:
            x = last_state

        # add bias
        if self.bias:
            u = np.hstack((1, single_input)).astype(self.typefloat)
        else:
            u = single_input

        # linear transformation
        x1 = np.dot(self.Win, u.reshape(self.shape[0], 1)) \
            + self.W.dot(x)

        # add feedback if requested
        if feedback is not None:
            x1 += feedback

        # previous states memory leak and non-linear transformation
        x1 = (1-self.leak)*x + self.leak*self.activation(x1)

        # return the next state computed
        return x1

    #? find a way to reinit everything when changing the seed
    #? even when the Initializers API is not used
    def reset_seed(self, seed: Union[int, RandomState]):
        if isinstance(seed, RandomState):
            self._rs = seed
        else:
            self._rs = RandomState(seed)
        self.seed = seed

        self._build()