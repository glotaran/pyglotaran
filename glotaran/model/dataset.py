"""This package contains glotaran's dataset class."""

from typing import Tuple
from copy import copy
import numpy as np


class Dataset:
    """Dataset is a small interface which dataset/-file implementations can
    fullfill in order to be consumable for glotaran.

    The Datasetclass is contains a very simple implementation used for
    simulatind data.
    """

    def __init__(self):
        self._axis = {}

    def get_axis(self, label: str) -> np.ndarray:
        """get_axis gets an axis by its label.

        Parameters
        ----------
        label : str
            The label of the axis.

        Returns
        -------
        axis : np.ndarray
        """
        return self._axis[label]

    def set_axis(self, label: str, axis: np.ndarray):
        """set_axis sets an axis by its label.

        Parameters
        ----------
        label : str
            The label of the axis.
        axis : np.ndarray
        """
        if not isinstance(axis, np.ndarray):
            raise TypeError("Axis must be of type numpy.ndarray")
        if any(not _hashable(v) for v in axis):
            raise ValueError("Axis elements must be hashable.")
        if isinstance(axis, list):
            axis = np.asarray(axis)
        self._axis[label] = axis

    def data(self) -> np.array:
        """Data returns the actual data of the dataset.

        Returns
        -------
        data : np.ndarray
        """
        return self._data

    def set_data(self, data: np.ndarray):
        """set_data sets the actual data of the dataset.

        Parameters
        ----------
        data : np.ndarray
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a nd array")
        if len(data.shape) is not 2:
            raise ValueError("Dataset must be 2-dimensional")
        self._data = data

    def copy(self) -> 'Dataset':
        """Returns a copy of the Dataset Object

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
