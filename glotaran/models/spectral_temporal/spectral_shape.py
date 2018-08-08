"""Glotaran Spectral Shape"""

from abc import ABC, abstractmethod
import numpy as np

from glotaran.model import ParameterGroup


class SpectralShape(ABC):
    """Base class for spectral shapes"""
    def __init__(self, label: str):
        """

        Parameters
        ----------
        label : str
            Label of the shape.


        Returns
        -------

        """
        self.label = label

    @abstractmethod
    def calculate(self, axis: np.ndarray, parameter: ParameterGroup) -> np.ndarray:
        """

        Parameters
        ----------
        axis: np.ndarray
            The axies to calculate the shape on.

        parameter: ParameterGroup
            The parameters to calculate the shape with.


        Returns
        -------
        shape: numpy.ndarray

        """
        raise NotImplementedError

    def __str__(self):
        """ """
        return f"* __{self.label}__"
