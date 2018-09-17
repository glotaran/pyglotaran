"""This package contains the spectral shape item."""

from typing import List
import numpy as np
from glotaran.model import model_item, model_item_typed


@model_item(attributes={
    'amplitude': List[str],
    'location': List[str],
    'width': List[str],
}, has_type=True)
class SpectralShapeGaussian:
    """A gaussian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Parameters
        ----------
        axis: np.ndarray
            The axies to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        matrix = self.amplitude * np.exp(
            -np.log(2) * np.square(2 * (axis - self.location)/self.width))
        print(matrix.shape)
        return matrix


@model_item(attributes={
}, has_type=True)
class SpectralShapeOne:
    """A gaussian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Parameters
        ----------
        axis: np.ndarray
            The axies to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        return np.ones((axis.shape[0]))


@model_item(attributes={
}, has_type=True)
class SpectralShapeZero:
    """A gaussian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Only works after calling calling 'fill'.

        Parameters
        ----------
        axis: np.ndarray
            The axies to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        return np.zeros((axis.shape[0]))


@model_item_typed(types={
    'gaussian': SpectralShapeGaussian,
    'one': SpectralShapeOne,
    'zero': SpectralShapeZero,
})
class SpectralShape:
    """Base class for spectral shapes"""
