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
        "skewness": {"type": Parameter, "allow_none": True},
    },
    has_type=True,
)
class SpectralShapeSkewedGaussian:
    """A (skewed) Gaussian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        r"""Calculate a (skewed) Gaussian shape for a given ``axis``.

        If a non-zero ``skewness`` parameter was added
        :func:`calculate_skewed_gaussian` will be used.
        Otherwise it will use :func:`calculate_gaussian`.

        Parameters
        ----------
        axis: np.ndarray
            The axis to calculate the shape for.

        Returns
        -------
        shape: numpy.ndarray
            A Gaussian shape.

        See Also
        --------
        calculate_gaussian
        calculate_skewed_gaussian

        Note
        ----
        Internally ``axis`` is converted from :math:`\mbox{nm}` to
        :math:`1/\mbox{cm}`, thus ``location`` and ``width`` also need to
        be provided in :math:`1/\mbox{cm}` (``1e7/value_in_nm``).

        """
        return (
            self.calculate_skewed_gaussian(axis)
            if self.skewness is not None and not np.allclose(self.skewness, 0)
            else self.calculate_gaussian(axis)
        )

    def calculate_gaussian(self, axis: np.ndarray) -> np.ndarray:
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
        return self.amplitude * np.exp(
            -np.log(2) * np.square(2 * ((1e7 / axis) - self.location) / self.width)
        )

    def calculate_skewed_gaussian(self, axis: np.ndarray) -> np.ndarray:
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
        see the definition in :func:`calculate_gaussian`.

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.


        Returns
        -------
        np.ndarray
            An array representing a skewed Gaussian shape.
        """
        log_args = 1 + (2 * self.skewness * ((1e7 / axis) - self.location) / self.width)
        result = np.zeros(log_args.shape)
        valid_arg_mask = np.where(log_args > 0)
        result[valid_arg_mask] = self.amplitude * np.exp(
            -np.log(2) * np.square(np.log(log_args[valid_arg_mask]) / self.skewness)
        )

        return result


@model_attribute(properties={}, has_type=True)
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


@model_attribute(properties={}, has_type=True)
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


@model_attribute_typed(
    types={
        "skewed-gaussian": SpectralShapeSkewedGaussian,
        "one": SpectralShapeOne,
        "zero": SpectralShapeZero,
    }
)
class SpectralShape:
    """Base class for spectral shapes"""
