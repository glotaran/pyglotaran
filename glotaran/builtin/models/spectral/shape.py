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
        "skew": {"type": Parameter, "allow_none": True},
    },
    has_type=True,
)
class SpectralShapeLorentzianBandshape:
    """A lorentzian spectral shape"""

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        r"""Calculate the lorentzian shape over ``axis``.

        If ``skew`` parameter was added and isn't close to zero
        :func:`calculate_skewed` will be used.
        Else it will use :func:`calculate_unskewed`.

        Parameters
        ----------
        axis: np.ndarray
            The axis to calculate the shape for.

        Returns
        -------
        shape: numpy.ndarray
            Skewed or unskewed lorentzian shape.

        See Also
        --------
        calculate_unskewed
        calculate_skewed

        Note
        ----
        Internally ``axis`` is converted from :math:`\mbox{nm}` to
        :math:`1/\mbox{cm}`, thus ``location`` and ``width`` also need to
        be provided in :math:`1/\mbox{cm}` (``1e7/value_in_nm``).

        """
        return (
            self.calculate_skewed(axis)
            if self.skew is not None and not np.allclose(self.skew, 0)
            else self.calculate_unskewed(axis)
        )

    def calculate_unskewed(self, axis: np.ndarray) -> np.ndarray:
        r"""Calcute the unskewed lorentzian shape for ``axis``.

        The following equation is used for the calculation:

        .. math::
            f(x, A, x_0, \sigma) = A \exp \left({-
            \frac{
                \log{\left(2 \right)
                \left(2(x - x_{0})\right)^{2}
            }}{\sigma^{2}}}\right)

        The parameters of the equation represent the following attributes of the shape:

        - :math:`x` :       ``axis``

        - :math:`A` :       ``amplitude``

        - :math:`x_0` :     ``location``

        - :math:`\sigma` :  ``width``

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.

        Returns
        -------
        np.ndarray
            Unskewed lorentzian shape.
        """
        return self.amplitude * np.exp(
            -np.log(2) * np.square(2 * ((1e7 / axis) - self.location) / self.width)
        )

    def calculate_skewed(self, axis: np.ndarray) -> np.ndarray:
        r"""Calcute the skewed lorentzian shape for ``axis``.

        The following equation is used for the calculation:

        .. math::
            f(x, x_0, A, \sigma, b) =
            \left\{
                \begin{array}{ll}
                    0                                               & \mbox{if } \theta \leq 0 \\
                    A \exp \left({- \dfrac{\log{\left(2 \right)}
                    \log{\left(\theta(x, x_0, \sigma, b) \right)}^{2}}{b^{2}}}\right)
                                                                    & \mbox{if } \theta > 0
                \end{array}
            \right.

        With:

        .. math::
            \theta(x, x_0, \sigma, b) = \frac{2 b \left(x - x_{0}\right) + \sigma}{\sigma}

        The parameters of the equation represent the following attributes of the shape:

        - :math:`x` :       ``axis``

        - :math:`A` :       ``amplitude``

        - :math:`x_0` :     ``location``

        - :math:`\sigma` :  ``width``

        - :math:`b` :       ``skew``

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.


        Returns
        -------
        np.ndarray
            Skewed lorentzian shape.
        """
        log_args = 1 + (2 * self.skew * ((1e7 / axis) - self.location) / self.width)
        result = np.zeros(log_args.shape)
        valid_arg_mask = np.where(log_args > 0)
        result[valid_arg_mask] = self.amplitude * np.exp(
            -np.log(2) * np.square(np.log(log_args[valid_arg_mask]) / self.skew)
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
        "lorentzian-bandshape": SpectralShapeLorentzianBandshape,
        "one": SpectralShapeOne,
        "zero": SpectralShapeZero,
    }
)
class SpectralShape:
    """Base class for spectral shapes"""
