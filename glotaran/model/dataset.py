"""Glotaran Dataset"""

from typing import Tuple
from copy import copy
import numpy as np


class Dataset:
    """Dataset encapsulates actual data and labeled axis. This class serves as
    base class for classes which wrap e.g. a CSV file.

    Parameters
    ----------
    label : Label of the Datasets

    Returns
    -------
    """

    def __init__(self):
        self._axis = {}
        self._estimated_axis = None
        self._calculated_axis = None

    def get_estimated_axis(self):
        """Get the axis along the calculated matrices are grouped"""
        return self._estimated_axis

    def set_estimated_axis(self, label: str):
        """Get the axis along the calculated matrices are grouped"""
        self._estimated_axis = label

    def get_calculated_axis(self):
        """Get the axis along the estimated matrices are grouped"""
        return self._calculated_axis

    def set_calculated_axis(self, label):
        """Get the axis along the estimated matrices are grouped"""
        self._calculated_axis = label

    def get_axis(self, label):
        """

        Parameters
        ----------
        label : label of the axis


        Returns
        -------
        axis: list or np.ndarray
        """
        return self._axis[label]

    def set_axis(self, label, axis):
        """

        Parameters
        ----------
        label : label of the axis

        axis: list or np.ndarray


        Returns
        -------

        """
        if not isinstance(axis, (list, np.ndarray)):
            raise TypeError("Axis must be list or ndarray")
        if any(not _hashable(v) for v in axis):
            raise ValueError("Axis must be list or ndarray of hashable values")
        if isinstance(axis, list):
            axis = np.asarray(axis)
        self._axis[label] = axis

    def get(self) -> np.array:
        """nd.array of shape (M,N)

        Returns
        -------
        data : np.array


        """
        return self._data

    def set(self, data):
        """

        Parameters
        ----------
        data : np.array


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

    def svd(self) -> Tuple[np.array, np.array, np.array]:
        """Returns the singular value decomposition of the dataset


        Returns
        -------
        tuple :
            (lsv, svals, rsv)

            lsv : np.array
                left singular values
            svals : np.array
                singular values
            rsv : np.array
                right singular values

        """
        lsv, svals, rsv = np.linalg.svd(self.get().T)
        return lsv, svals, rsv.T


def _hashable(value):
    try:
        # pylint: disable=pointless-statement
        {value: None}
        return True
    except Exception:
        return False
