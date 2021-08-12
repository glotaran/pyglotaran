"""This package contains the spectral shape item."""

import numpy as np

from glotaran.model import model_item
from glotaran.model import model_item_typed
from glotaran.parameter import Parameter


@model_item(
    properties={
        "amplitude": {"type": Parameter, "allow_none": True},
        "location": Parameter,
        "width": Parameter,
    },
    has_type=True,
)
class SpectralShapeGaussian:
    """A Gaussian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        r"""Calculate a normal Gaussian shape for a given ``axis``.

        The following equation is used for the calculation:

        .. math::
            f(x, A, x_0, \Delta) = A \exp \left({-
            \frac{
                \log{\left(2 \right)
                \left(2(x - x_{0})\right)^{2}
            }}{\Delta^{2}}}\right)

        The parameters of the equation represent the following attributes of the shape:

        - :math:`x` :       ``axis``

        - :math:`A` :       ``amplitude``

        - :math:`x_0` :     ``location``

        - :math:`\Delta` :  ``width``

        In this formalism, :math:`\Delta` represents the full width at half maximum (FWHM).
        Compared to the more common definition
        :math:`\exp \left(- (x-\mu )^{2}/(2\sigma^{2})\right)`
        we have :math:`\sigma = \Delta/(2\sqrt{2\ln(2)})=\Delta/2.35482`

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.

        Returns
        -------
        np.ndarray
            An array representing a Gaussian shape.
        """
        shape = np.exp(-np.log(2) * np.square(2 * (axis - self.location) / self.width))
        if self.amplitude is not None:
            shape *= self.amplitude
        return shape


@model_item(
    properties={
        "skewness": Parameter,
    },
    has_type=True,
)
class SpectralShapeSkewedGaussian(SpectralShapeGaussian):
    """A skewed Gaussian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        r"""Calculate the skewed Gaussian shape for ``axis``.

        The following equation is used for the calculation:

        .. math::
            f(x, x_0, A, \Delta, b) =
            \left\{
                \begin{array}{ll}
                    0                                               & \mbox{if } \theta \leq 0 \\
                    A \exp \left({- \dfrac{\log{\left(2 \right)}
                    \log{\left(\theta(x, x_0, \Delta, b) \right)}^{2}}{b^{2}}}\right)
                                                                    & \mbox{if } \theta > 0
                \end{array}
            \right.

        With:

        .. math::
            \theta(x, x_0, \Delta, b) = \frac{2 b \left(x - x_{0}\right) + \Delta}{\Delta}

        The parameters of the equation represent the following attributes of the shape:

        - :math:`x` :       ``axis``

        - :math:`A` :       ``amplitude``

        - :math:`x_0` :     ``location``

        - :math:`\Delta` :  ``width``

        - :math:`b` :       ``skewness``

        Where :math:`\Delta` represents the full width at half maximum (FWHM),
        see :func:`calculate_gaussian`.

        Note that in the limit of skewness parameter :math:`b` equal to zero
        :math:`f(x, x_0, A, \Delta, b)` simplifies to a normal gaussian
        (since :math:`\lim_{b \to 0} \frac{\ln(1+bx)}{b}=x`),
        see the definition in :func:`SpectralShapeGaussian.calculate`.

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.


        Returns
        -------
        np.ndarray
            An array representing a skewed Gaussian shape.
        """
        if np.allclose(self.skewness, 0):
            return super().calculate(axis)
        log_args = 1 + (2 * self.skewness * (axis - self.location) / self.width)
        shape = np.zeros(log_args.shape)
        valid_arg_mask = np.where(log_args > 0)
        shape[valid_arg_mask] = np.exp(
            -np.log(2) * np.square(np.log(log_args[valid_arg_mask]) / self.skewness)
        )
        if self.amplitude is not None:
            shape *= self.amplitude
        return shape


@model_item(properties={}, has_type=True)
class SpectralShapeOne:
    """A constant spectral shape with value 1"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Parameters
        ----------
        axis: np.ndarray
            The axis to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        return np.ones(axis.shape[0])


@model_item(properties={}, has_type=True)
class SpectralShapeZero:
    """A constant spectral shape with value 0"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Only works after calling ``fill``.

        Parameters
        ----------
        axis: np.ndarray
            The axis to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        return np.zeros(axis.shape[0])


@model_item_typed(
    types={
        "gaussian": SpectralShapeGaussian,
        "skewed-gaussian": SpectralShapeSkewedGaussian,
        "one": SpectralShapeOne,
        "zero": SpectralShapeZero,
    }
)
class SpectralShape:
    """Base class for spectral shapes"""
