""" Glotaran Gaussian Spectral Shape """

import numpy as np

from glotaran.model import ParameterGroup

from .spectral_shape import SpectralShape


class SpectralShapeGaussian(SpectralShape):
    """A gaussian spectral shape"""
    def __init__(self, label: str, amplitude: str, location: str, width: str):
        """

        Parameters
        ----------
        label: str
            The label of the shape.

        amplitude: str
            The amplitude parameter.

        location: str
            The location parameter.

        width: str
            The width parameter.


        """
        super(SpectralShapeGaussian, self).__init__(label)
        self.amplitude = amplitude
        self.location = location
        self.width = width

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
        amp = parameter.get(self.amplitude)
        location = parameter.get(self.location)
        width = parameter.get(self.width)
        return amp * np.exp(-np.log(2) *
                            np.square(2 * (axis - location)/width))

    def __str__(self):
        string = super(SpectralShapeGaussian, self).__str__()
        string += ", _Type_: Gaussian"
        string += f", _Amplitude_: {self.amplitude}"
        string += f", _Location_: {self.location}"
        string += f", _Width_: {self.width}"
        return string
