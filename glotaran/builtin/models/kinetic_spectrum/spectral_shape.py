"""This package contains the spectral shape item."""

import numpy as np

from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        "amplitude": Parameter,
        "location": Parameter,
        "width": Parameter,
    },
    has_type=True,
)
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
        return self.amplitude * np.exp(
            -np.log(2) * np.square(2 * (axis - self.location) / self.width)
        )


@model_attribute(properties={}, has_type=True)
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
        return np.ones(axis.shape[0])


@model_attribute(properties={}, has_type=True)
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
        return np.zeros(axis.shape[0])


@model_attribute_typed(
    types={
        "gaussian": SpectralShapeGaussian,
        "one": SpectralShapeOne,
        "zero": SpectralShapeZero,
    }
)
class SpectralShape:
    """Base class for spectral shapes"""
