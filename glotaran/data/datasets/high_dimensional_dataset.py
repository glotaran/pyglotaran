"""
Module with specialized dataset subclasses.
The datasets in this module are specialized versions of the general dataset,
since some kinds of data need additional information to processed, viewed or
mapped back properly.
"""
from typing import Callable, List, Tuple, Union

import numpy as np

from glotaran.model.dataset import Dataset


class HighDimensionalDataset(Dataset):
    """
    Baseclass for high dimensional data like FLIM
    """

    def __init__(self,
                 mapper: Callable[[np.ndarray],
                                  Union[List[Union[list, tuple]],
                                        np.ndarray,
                                        Tuple[Union[list, tuple]]]],
                 orig_shape: tuple):
        """

        Parameters
        ----------

        mapper
        orig_shape
        """
        super().__init__()
        self._mapper = mapper
        self._orig_shape = orig_shape

    @property
    def orig_shape(self) -> tuple:
        """
        Original shape of the high dimensional data.

        Returns
        -------
        orig_shape: tuple
            Original shape of the high dimensional data.
        """
        return self._orig_shape

    @orig_shape.setter
    def orig_shape(self, orig_shape):
        if isinstance(orig_shape, tuple) and len(orig_shape):
            self._orig_shape = orig_shape
        elif not isinstance(orig_shape, tuple):
            raise TypeError("orig_shape needs to be of type tuple")
        else:
            raise ValueError("orig_shape needs to be not empty")

    @property
    def mapper(self) -> Callable[[np.ndarray],
                                 Union[List[Union[list, tuple]],
                                       np.ndarray,
                                       Tuple[Union[list, tuple]]]]:
        """
        Mapper function/method which is used to to reduce the
        dimensionality of the high dimensional data.

        Returns
        -------
        mapper: Callable

        """
        return self._mapper

    @mapper.setter
    def mapper(self,
               mapper: Callable[[np.ndarray],
                                Union[List[Union[list, tuple]],
                                      np.ndarray,
                                      Tuple[Union[list, tuple]]]]):
        if isinstance(mapper, Callable):
            self._mapper = mapper
        else:
            raise TypeError("`mapper` needs to be a function or method.")
