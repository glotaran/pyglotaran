"""This package contains irf items."""


import numpy as np

from glotaran.model import ModelError
from glotaran.model import ModelItemTyped
from glotaran.model import ParameterType
from glotaran.model import attribute
from glotaran.model import item


@item
class Irf(ModelItemTyped):
    """Represents an IRF."""


@item
class IrfMultiGaussian(Irf):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameters
    ----------

    label:
        label of the irf
    center:
        one or more center of the irf as parameter indices
    width:
        one or more widths of the gaussian as parameter index
    center_dispersion_coefficients:
        polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion_coefficients:
        polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """

    type: str = "multi-gaussian"

    center: list[ParameterType]
    width: list[ParameterType]
    scale: list[ParameterType] | None = None
    shift: list[ParameterType] | None = None
    normalize: bool = True
    backsweep: bool = False
    backsweep_period: ParameterType | None = None

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
        """Returns the properties of the irf with shift applied."""

        centers = self.center if isinstance(self.center, list) else [self.center]
        centers = np.asarray([c.value for c in centers])

        widths = self.width if isinstance(self.width, list) else [self.width]
        widths = np.asarray([w.value for w in widths])

        len_centers = len(centers)
        len_widths = len(widths)
        if len_centers != len_widths:
            if min(len_centers, len_widths) != 1:
                raise ModelError(
                    f"len(centers) ({len_centers}) not equal "
                    f"len(widths) ({len_widths}) none of is 1."
                )
            if len_centers == 1:
                centers = np.asarray([centers[0] for _ in range(len_widths)])
            else:
                widths = np.asarray([widths[0] for _ in range(len_centers)])

        scales = self.scale if self.scale is not None else [1.0 for _ in centers]
        scales = scales if isinstance(scales, list) else [scales]
        scales = np.asarray(scales)

        shift = 0
        if self.shift is not None:
            if global_index >= len(self.shift):
                raise ModelError(
                    f"No shift parameter for index {global_index} "
                    f"({global_axis[global_index]}) in irf {self.label}"
                )
            shift = self.shift[global_index]

        backsweep = self.backsweep

        backsweep_period = self.backsweep_period.value if self.backsweep else 0

        return centers, widths, scales, shift, backsweep, backsweep_period

    def calculate(self, index: int, global_axis: np.ndarray, model_axis: np.ndarray) -> np.ndarray:
        centers, widths, scales, _, _, _ = self.parameter(index, global_axis)
        return sum(
            scale * np.exp(-1 * (model_axis - center) ** 2 / (2 * width**2))
            for center, width, scale in zip(centers, widths, scales)
        )

    def is_index_dependent(self):
        return self.shift is not None


@item
class IrfGaussian(IrfMultiGaussian):
    type: str = "gaussian"
    center: ParameterType
    width: ParameterType


@item
class IrfSpectralMultiGaussian(IrfMultiGaussian):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameters
    ----------

    label:
        label of the irf
    center:
        one or more center of the irf as parameter indices
    width:
        one or more widths of the gaussian as parameter index
    center_dispersion_coefficients:
        list of parameters with polynomial coefficients describing
        the dispersion of the irf center location. None for no dispersion.
    width_dispersion_coefficients:
        list of parameters with polynomial coefficients describing
        the dispersion of the width of the irf. None for no dispersion.

    """

    type: str = "spectral-multi-gaussian"
    dispersion_center: ParameterType
    center_dispersion_coefficients: list[ParameterType]
    width_dispersion_coefficients: list[ParameterType] = attribute(factory=list)
    model_dispersion_with_wavenumber: bool = False

    def parameter(self, global_index: int, global_axis: np.ndarray):
        """Returns the properties of the irf with shift and dispersion applied."""
        centers, widths, scale, shift, backsweep, backsweep_period = super().parameter(
            global_index, global_axis
        )

        index = global_axis[global_index] if global_index is not None else None

        if self.dispersion_center is not None:
            dist = (
                (1e3 / index - 1e3 / self.dispersion_center)
                if self.model_dispersion_with_wavenumber
                else (index - self.dispersion_center) / 100
            )

        if len(self.center_dispersion_coefficients) != 0:
            if self.dispersion_center is None:
                raise ModelError(f"No dispersion center defined for irf '{self.label}'")
            for i, disp in enumerate(self.center_dispersion_coefficients):
                centers += disp * np.power(dist, i + 1)

        if len(self.width_dispersion_coefficients) != 0:
            if self.dispersion_center is None:
                raise ModelError(f"No dispersion center defined for irf '{self.label}'")
            for i, disp in enumerate(self.width_dispersion_coefficients):
                widths = widths + disp * np.power(dist, i + 1)

        return centers, widths, scale, shift, backsweep, backsweep_period

    def calculate_dispersion(self, axis):
        dispersion = []
        for index, _ in enumerate(axis):
            center, _, _, _, _, _ = self.parameter(index, axis)
            dispersion.append(center)
        return np.asarray(dispersion).T

    def is_index_dependent(self):
        return super().is_index_dependent() or self.dispersion_center is not None


@item
class IrfSpectralGaussian(IrfSpectralMultiGaussian):
    type: str = "spectral-gaussian"
    center: ParameterType
    width: ParameterType
