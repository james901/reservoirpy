"""
===================
Weight initializers
===================
"""

import warnings

from typing import Union

from numpy.random import RandomState

from ._base import Initializer
from ._base import RandomSparse

from ._internal import FastSpectralScaling
from ._internal import SpectralScaling
from ._internal import NormalSpectralScaling
from ._internal import UniformSpectralScaling
from ._internal import BinarySpectralScaling

from ._input_feedback import NormalScaling
from ._input_feedback import UniformScaling
from ._input_feedback import BinaryScaling


__all__ = [
    "initializer", "Initializer", "RandomSparse",
    "FastSpectralScaling", "SpectralScaling",
    "NormalSpectralScaling", "UniformSpectralScaling",
    "BinarySpectralScaling", "NormalScaling", "UniformScaling",
    "BinaryScaling"
]


_registry = {
        "normal": {
            "spectral": NormalSpectralScaling,
            "scaling": NormalScaling
        },
        "uniform": {
            "spectral": UniformSpectralScaling,
            "scaling": UniformScaling
        },
        "binary": {
            "spectral": BinarySpectralScaling,
            "scaling": BinaryScaling
        },
        "fsi": FastSpectralScaling
    }


def initializer(method: str,
                connectivity: float = None,
                scaling: int = None,
                spectral_radius: int = None,
                seed: Union[int, RandomState] = None,
                **kwargs
                ) -> Initializer:
    """Returns an intializer given the
    parameters.

    Parameters
    ----------
    method : {"normal", "uniform", "binary", "fsi"}
        Method used for randomly sample the weights.
        "fsi" can only be used with spectral scaling.
    connectivity : float, optional
        Probability of connection between units. Density of
        the sparse matrix.
    scaling : int, optional
        Scaling coefficient to apply on the weights. Can not be used
        with spectral scaling.
    spectral_radius : int, optional
        Maximum eigenvalue of the initialized matrix. Can not be used
        with regular scaling.
    seed : int or RandomState, optional
        Random state generator seed or RandomState instance.

    Returns
    -------
    Initializer
        An :class: `Initializer` object.

    Raises
    ------
    ValueError
        Can't perfom both spectral scaling and regular scaling.
    ValueError
        Method is not recognized.
    """
    if scaling is not None and spectral_radius is not None:
        raise ValueError("can't perfom both spectral scaling and regular scaling.")

    selection = _registry.get(method)

    if selection is None:
        raise ValueError(f"'{method}' is not a valid method. "
                         "Must be 'fsi', 'normal', 'uniform' or 'binary'.")

    if method == "fsi":
        return selection(spectral_radius=spectral_radius,
                         connectivity=connectivity,
                         seed=seed,
                         **kwargs)

    if scaling is None:
        if spectral_radius is not None:
            selected_initializer = selection["spectral"]
            return selected_initializer(spectral_radius=spectral_radius,
                                        connectivity=connectivity,
                                        seed=seed,
                                        **kwargs)
        else:
            warnings.warn("neither 'spectral_radius' nor 'scaling' are "
                          "set. Default initializer returned will then be "
                          "a constant scaling initializer.", UserWarning)

    selected_initializer = selection["scaling"]
    return selected_initializer(scaling=scaling,
                                connectivity=connectivity,
                                seed=seed,
                                **kwargs)
