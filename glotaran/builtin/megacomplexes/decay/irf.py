"""This package contains irf items."""

from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import ModelError
from glotaran.model import model_item
from glotaran.model import model_item_typed
from glotaran.parameter import Parameter


@model_item(has_type=True)
class IrfMeasured:
    """A measured IRF. The data must be supplied by the dataset."""


@model_item(
    properties={
        "center": List[Parameter],
        "width": List[Parameter],
        "scale": {"type": List[Parameter], "allow_none": True},
        "shift": {"type": List[Parameter], "allow_none": True},
        "normalize": {"type": bool, "default": True},
        "backsweep": {"type": bool, "default": False},
        "backsweep_period": {"type": Parameter, "allow_none": True},
    },
    has_type=True,
)
class IrfMultiGaussian:
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

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
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
                centers = [centers[0] for _ in range(len_widths)]
            else:
                widths = [widths[0] for _ in range(len_centers)]

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
            scale * np.exp(-1 * (model_axis - center) ** 2 / (2 * width ** 2))
            for center, width, scale in zip(centers, widths, scales)
        )

    def is_index_dependent(self):
        return self.shift is not None


@model_item(
    properties={
        "center": Parameter,
        "width": Parameter,
    },
    has_type=True,
)
class IrfGaussian(IrfMultiGaussian):
    pass


@model_item(
    properties={
        "dispersion_center": {"type": Parameter, "allow_none": True},
        "center_dispersion_coefficients": {"type": List[Parameter], "default": []},
        "width_dispersion_coefficients": {"type": List[Parameter], "default": []},
        "model_dispersion_with_wavenumber": {"type": bool, "default": False},
    },
    has_type=True,
)
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


@model_item(
    properties={
        "center": Parameter,
        "width": Parameter,
    },
    has_type=True,
)
class IrfSpectralGaussian(IrfSpectralMultiGaussian):
    pass


@model_item_typed(
    types={
        "gaussian": IrfGaussian,
        "multi-gaussian": IrfMultiGaussian,
        "spectral-multi-gaussian": IrfSpectralMultiGaussian,
        "spectral-gaussian": IrfSpectralGaussian,
    }
)
class Irf:
    """Represents an IRF."""
