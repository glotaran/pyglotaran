"""
Module with specialized dataset subclasses.
The datasets in this module are specialized versions of the general dataset,
since some kinds of data need additional information to processed, viewed or
mapped back properly.
"""
from typing import Callable, List, Tuple, Union

import numpy as np

from .dataset import Dataset


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


class FLIMDataset(HighDimensionalDataset):
    """
    Custom Data set for FLIM data
    """

    _time_units = {
        "h":   3600,
        "m":     60,
        "s":      1,
        "ms":  1e-3,
        "us":  1e-6,
        "ns":  1e-9,
        "ps":  1e-12,
        "fs":  1e-15,
    }

    def __init__(self,
                 mapper: Callable[[np.ndarray],
                                  Union[List[tuple],
                                        Tuple[tuple],
                                        np.ndarray]],
                 orig_shape: tuple, time_unit: str="s"):
        """

        Parameters
        ----------
        mapper: Callable[[np.ndarray],  Union[List[tuple],
        Tuple[tuple], np.ndarray ]]

        orig_shape: tuple
        """
        super().__init__(mapper, orig_shape)
        self._intensity_map = None
        self.mapper = mapper
        self.orig_shape = orig_shape
        self.time_unit = time_unit

    @property
    def time_unit(self) -> str:
        """the time unit [default 's'] """
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value: str):
        if value not in self._time_units:
            raise ValueError(f'Unknown time unit {value}. Supported units are '
                             f'{",".join(self._time_units.keys())}')
        self._time_unit = value

    @property
    def time_axis(self) -> np.ndarray:
        """
        Time axis of the data

        Returns
        -------
        time_axis: np.ndarray

        """
        return self.get_axis("time")

    @time_axis.setter
    def time_axis(self, value: Union[List, Tuple, np.ndarray]):
        self.set_axis("time", value)

    @property
    def pixel_axis(self) -> Union[List[tuple], Tuple[tuple], np.ndarray]:
        """
        Pixel coordinates of the time traces, which were mapped
        from the high dimensional data

        Returns
        -------
        pixel_axis: Union[List[tuple], Tuple[tuple], np.ndarray]
        """
        return self.get_axis("pixel")

    @pixel_axis.setter
    def pixel_axis(self, value: Union[List[tuple], Tuple[tuple], np.ndarray]):
        self.set_axis("pixel", value)

    @property
    def intensity_map(self) -> np.ndarray:
        """
        Intensity map (pixel map with sum over time traces as values)
        of the FLIM data

        Returns
        -------
        intensity_map: np.ndarray
        """
        return self._intensity_map

    @intensity_map.setter
    def intensity_map(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("The intensity_map needs to be a ndarray")
        if len(value.shape) is not 2:
            raise ValueError("The intensity_map needs to be 2-dimensional")
        self._intensity_map = value

    def get_estimated_axis(self):
        """ """
        return self.pixel_axis

    def get_calculated_axis(self):
        """ """
        return self.time_axis
