import typing

import numpy as np

from glotaran.builtin.models.kinetic_image.irf import Irf
from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.model import model_attribute
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        "dispersion_center": {"type": Parameter, "allow_none": True},
        "center_dispersion": {"type": typing.List[Parameter], "default": []},
        "width_dispersion": {"type": typing.List[Parameter], "default": []},
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
    center_dispersion:
        polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion:
        polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """

    def parameter(self, global_index: int, global_axis: np.ndarray):
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

        if len(self.center_dispersion) != 0:
            if self.dispersion_center is None:
                raise Exception(self, f'No dispersion center defined for irf "{self.label}"')
            for i, disp in enumerate(self.center_dispersion):
                centers += disp * np.power(dist, i + 1)

        if len(self.width_dispersion) != 0:
            if self.dispersion_center is None:
                raise Exception(self, f'No dispersion center defined for irf "{self.label}"')
            for i, disp in enumerate(self.width_dispersion):
                widths = widths + disp * np.power(dist, i + 1)

        return centers, widths, scale, shift, backsweep, backsweep_period

    def calculate_dispersion(self, axis):
        dispersion = []
        for index, _ in enumerate(axis):
            center, _, _, _, _, _ = self.parameter(index, axis)
            dispersion.append(center)
        return np.asarray(dispersion).T


@model_attribute(
    properties={
        "center": Parameter,
        "width": Parameter,
    },
    has_type=True,
)
class IrfSpectralGaussian(IrfSpectralMultiGaussian):
    pass


Irf.add_type("spectral-multi-gaussian", IrfSpectralMultiGaussian)
Irf.add_type("spectral-gaussian", IrfSpectralGaussian)
