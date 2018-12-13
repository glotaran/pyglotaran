"""
Module with specialized dataset subclasses.
The datasets in this module are specialized versions of the general dataset,
since some kinds of data need additional information to processed, viewed or
mapped back properly.
"""
from typing import Callable, List, Tuple, Union

import numpy as np

from .high_dimensional_dataset import HighDimensionalDataset


def get_pixel_map(array: Union[None, np.ndarray] = None,
                  shape: Union[set, tuple, list] = None,
                  x_index: int = 0, y_index: int = 1,
                  transposed: bool = False)\
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
                 orig_shape: tuple, time_unit: str = "s"):
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
    def pixel_axis_x(self):
        axis = []
        for pixel in self.pixel_axis:
            if pixel[0] not in axis:
                axis.append(pixel[0])
        return axis

    @property
    def pixel_axis_y(self):
        axis = []
        for pixel in self.pixel_axis:
            if pixel[1] not in axis:
                axis.append(pixel[1])
        return axis

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

    def cut(self,
            upper_left: Tuple[int, int],
            lower_right: Tuple[int, int],
            transform_pixel: bool = False,
            ) -> "FLIMDataset":
        """
        Returns a FLIMDataset containing only the timetraces in the rectangle
        defined by the upper left and the lower right pixel.

        Parameters
        ----------
        upper_left : Tuple[int, int]
            Upper Left Pixel
        lower_right : Tuple[int, int]
            Lower Right Pixel
        transform_pixel: bool, default False
            Set to True to transform the pixel coordinates to new origin.

        Returns
        -------
        cutted_dataset: FLIMDataset
        """
        indices = []
        for i, (x, y) in enumerate(self.pixel_axis):
            if upper_left[0] <= x < lower_right[0] and lower_right[1] <= y < upper_left[1]:
                indices.append(i)

        data = self.data()
        cut_data = data[:, indices]

        time = self.get_axis("time")
        orig_shape = (lower_right[0] - upper_left[0], upper_left[1] - lower_right[1], time.size)
        flim_dataset = FLIMDataset(get_pixel_map, orig_shape, self.time_unit)

        pixel = self.pixel_axis[indices]

        if transform_pixel:
            for i, pxl in enumerate(pixel):
                pixel[i] = (pxl[0] - upper_left[0], pxl[1] - lower_right[1])

        flim_dataset.set_axis('time', time)
        flim_dataset.set_axis('pixel', pixel)
        flim_dataset.set_data(cut_data)

        intensity_map = cut_data.reshape(orig_shape).sum(axis=2)
        flim_dataset.intensity_map = intensity_map
        return flim_dataset
