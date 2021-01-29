import typing

import numba as nb
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

    def parameter(self, index):
        centers, widths, scale, backsweep, backsweep_period = super().parameter(index)

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

        return centers, widths, scale, backsweep, backsweep_period

    def calculate_dispersion(self, axis):
        dispersion = []
        for index in axis:
            center, _, _, _, _ = self.parameter(index)
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


@model_attribute(
    properties={
        "coherent_artifact_order": {"type": int},
        "coherent_artifact_width": {"type": Parameter, "allow_none": True},
    },
    has_type=True,
)
class IrfGaussianCoherentArtifact(IrfSpectralGaussian):
    def clp_labels(self):
        return [f"coherent_artifact_{i}" for i in range(1, self.coherent_artifact_order + 1)]

    def calculate_coherent_artifact(self, axis):
        if not 1 <= self.coherent_artifact_order <= 3:
            raise Exception(self, "Coherent artifact order must be between in [1,3]")

        center, width, _, _, _ = self.parameter(None)

        center = center[0]
        width = (
            self.coherent_artifact_width.value
            if self.coherent_artifact_width is not None
            else width[0]
        )

        clp_label = self.clp_labels()

        matrix = self._calculate_coherent_artifact_matrix(
            center, width, axis, self.coherent_artifact_order
        )

        return clp_label, matrix

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _calculate_coherent_artifact_matrix(center, width, axis, order):
        matrix = np.zeros((axis.size, order), dtype=np.float64)

        matrix[:, 0] = np.exp(-1 * (axis - center) ** 2 / (2 * width ** 2))
        if order > 1:
            matrix[:, 1] = matrix[:, 0] * (center - axis) / width ** 2

        if order > 2:
            matrix[:, 2] = (
                matrix[:, 0]
                * (center ** 2 - width ** 2 - 2 * center * axis + axis ** 2)
                / width ** 4
            )
        return matrix


Irf.add_type("spectral-multi-gaussian", IrfSpectralMultiGaussian)
Irf.add_type("spectral-gaussian", IrfSpectralGaussian)
Irf.add_type("gaussian-coherent-artifact", IrfGaussianCoherentArtifact)
