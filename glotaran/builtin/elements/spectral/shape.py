"""This package contains the spectral shape item."""

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class SpectralShape(TypedItem):
    pass


class SpectralShapeGaussian(SpectralShape):
    """A Gaussian spectral shape"""

    type: Literal["gaussian"]  # type:ignore[assignment]
    amplitude: ParameterType | None = None
    location: ParameterType
    width: ParameterType

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
        shape = np.exp(
            -np.log(2)
            * np.square(2 * (axis - self.location) / self.width)  # type:ignore[operator]
        )
        if self.amplitude is not None:
            shape *= self.amplitude
        return shape


class SpectralShapeSkewedGaussian(SpectralShapeGaussian):
    """A skewed Gaussian spectral shape"""

    type: Literal["skewed-gaussian"]  # type:ignore[assignment]
    skewness: ParameterType

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
        log_args: ArrayLike = 1 + (  # type:ignore[assignment]
            2 * self.skewness * (axis - self.location) / self.width  # type:ignore[operator]
        )
        shape = np.zeros(log_args.shape)
        valid_arg_mask = np.where(log_args > 0)
        shape[valid_arg_mask] = np.exp(
            -np.log(2) * np.square(np.log(log_args[valid_arg_mask]) / self.skewness)
        )
        if self.amplitude is not None:
            shape *= self.amplitude  # type:ignore[arg-type]
        return shape


class SpectralShapeOne(SpectralShape):
    """A constant spectral shape with value 1"""

    type: Literal["one"]  # type:ignore[assignment]

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


class SpectralShapeZero(SpectralShape):
    """A constant spectral shape with value 0"""

    type: Literal["zero"]  # type:ignore[assignment]

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
