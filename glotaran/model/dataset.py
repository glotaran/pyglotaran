"""Glotaran Dataset"""

from copy import copy
from typing import Union

import numpy as np


class DimensionalityError(Exception):
    """
    Custom exception if data have the wrong dimensionality
    """
    pass


class Dataset:
    """Dataset is a small interface which dataset/-file implementations can
    fullfill in order to be consumable for glotaran.

    The Datasetclass is contains a very simple implementation used for
    simulatind data.
    """

    def __init__(self):
        self._axis = {}

    def get_axis(self, axis_label: str) -> np.ndarray:
        """get_axis gets an axis by its axis_label.

        Parameters
        ----------
        axis_label : str
            The label of the axis.

        Returns
        -------
        axis : np.ndarray
        """
        return self._axis[axis_label]

    def set_axis(self, axis_label: str, axis:  Union[list, tuple, np.ndarray]):
        """

        Parameters
        ----------
        axis_label: str
            label of the axis

        axis: list or np.ndarray
        """
        if not isinstance(axis, (list, tuple, np.ndarray)):
            raise TypeError(f"Axis must be list, tuple or ndarray, got {type(axis)} instead.")
        if any(not _hashable(v) for v in axis):
            raise ValueError("Axis elements must be hashable.")
        if isinstance(axis, (list, tuple)):
            axis = np.asarray(axis)
        self._axis[axis_label] = axis

    def data(self) -> np.ndarray:
        """
        Data of the dataset as np.ndarray of shape (M,N).
        This corresponds to a measurement with M indices (i.e. wavelength, pixel)
        and N time steps.

        Returns
        -------
        data: np.ndarray
            Data of the dataset as np.ndarray of shape (M,N)
        """
        return self._data

    def set_data(self, data: np.ndarray):
        """
        Data of the dataset as np.ndarray of shape (M,N).
        This corresponds to a measurement with M indices (i.e. wavelength, pixel)
        and N time steps.

        Parameters
        ----------
        data: np.ndarray
            Data of the dataset as np.ndarray of shape (M,N)

        """
        if not isinstance(data, np.ndarray):
            raise TypeError("The data needs to be a ndarray")
        if len(data.shape) is not 2:
            raise DimensionalityError("The data needs to be 2-dimensional")
        self._data = data

    def copy(self) -> 'Dataset':
        """
        Returns a copy of the Dataset Object

        Returns
        -------
        dataset : Dataset
        """
        return copy(self)


def _hashable(value):
    try:
        # pylint: disable=pointless-statement
        {value: None}
        return True
    except Exception:
        return False
