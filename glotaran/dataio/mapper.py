"""
This module contains mapping functions
i.e. to map FLIM data to pixel position tuples
"""
from typing import Union, Tuple

import numpy as np


def get_pixel_map(array: Union[None, np.ndarray]=None,
                  shape: Union[set, tuple, list]=None,
                  x_index: int=0, y_index: int=1,
                  transposed: bool=False)\
        -> Union[Tuple[tuple], Tuple[tuple, tuple]]:
    """

    Parameters
    ----------
    array: np.ndarray, default: None
    shape: set, tuple, list, default: None
    x_index: int

    y_index: int
    transposed: bool
        False:
            A n-tuple of 2-tuples is returned, with can be used as axis for a dataset
        True:
            A 2-tuple of n-tuples is returned, wich can be used to index a np.ndarray
            of the given shape

    Returns
    -------

    """
    if array is not None and shape is not None:
        raise ValueError("Using both, `array` and `shape`, is ambiguous. "
                         "Please only use one of them.")
    if isinstance(array, np.ndarray):
        if len(array.shape) is 3:
            grid = np.indices(array.shape)
        else:
            raise ValueError(f"This mapper is designed to map 3 dimensional "
                             f"data (len(array.shape)=3) to its pixel map (x-y-plane). "
                             f"The shape you provided has a dimensionality of {len(array.shape)}.")
    elif isinstance(shape, (set, tuple, list)):
        if len(shape) is 3:
            grid = np.indices(shape)
        else:
            raise ValueError(f"This mapper is designed to map 3 dimensional "
                             f"data (len(shape)=3) to its pixel map (x-y-plane). "
                             f"The shape you provided has a dimensionality of {len(shape)}.")
    else:
        raise ValueError("You to provide either an `array` or a `shape` of dimension 3.")

    x_indices = tuple(grid[x_index][:, :, 0].ravel())
    y_indices = tuple(grid[y_index][:, :, 0].ravel())
    if transposed:
        return x_indices, y_indices
    else:
        return tuple(map(tuple, np.array([x_indices, y_indices]).T))
