"""This package contains irf items."""

from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import ModelError
from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed
from glotaran.parameter import Parameter


@model_attribute(has_type=True)
class IrfMeasured:
    """A measured IRF. The data must be supplied by the dataset."""


@model_attribute(
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
    center_dispersion:
        polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion:
        polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:

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
                len_centers = len_widths
            else:
                widths = [widths[0] for _ in range(len_centers)]
                len_widths = len_centers

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


@model_attribute(
    properties={
        "center": Parameter,
        "width": Parameter,
    },
    has_type=True,
)
class IrfGaussian(IrfMultiGaussian):
    pass


@model_attribute_typed(
    types={
        "gaussian": IrfGaussian,
        "multi-gaussian": IrfMultiGaussian,
        "measured": IrfMeasured,
    }
)
class Irf:
    """Represents an IRF."""
