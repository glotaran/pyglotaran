"""This package contains the kinetic megacomplex item."""
from __future__ import annotations

import numba as nb
import numpy as np

from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import model_attribute
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        "order": {"type": int},
        "width": {"type": Parameter, "allow_none": True},
    },
)
class CoherentArtifactMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        model,
        dataset_descriptor: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        if not 1 <= self.order <= 3:
            raise ModelError("Coherent artifact order must be between in [1,3]")

        if dataset_descriptor.irf is None:
            raise ModelError(f'No irf in dataset "{dataset_descriptor.label}"')

        if not isinstance(dataset_descriptor.irf, IrfMultiGaussian):
            raise ModelError(f'Irf in dataset "{dataset_descriptor.label} is not a gaussian irf."')

        global_index = indices.get(model.global_dimension, None)
        global_axis = axis.get(model.global_dimension, None)
        irf = dataset_descriptor.irf

        center, width, _, _, _, _ = irf.parameter(global_index, global_axis)
        center = center[0]
        width = self.width.value if self.width is not None else width[0]

        axis = axis[model.model_dimension]
        matrix = _calculate_coherent_artifact_matrix(center, width, axis, self.order)
        return (self.compartments(), matrix)

    def compartments(self):
        return [f"coherent_artifact_{i}" for i in range(1, self.order + 1)]


@nb.jit(nopython=True, parallel=True)
def _calculate_coherent_artifact_matrix(center, width, axis, order):
    matrix = np.zeros((axis.size, order), dtype=np.float64)

    matrix[:, 0] = np.exp(-1 * (axis - center) ** 2 / (2 * width ** 2))
    if order > 1:
        matrix[:, 1] = matrix[:, 0] * (center - axis) / width ** 2

    if order > 2:
        matrix[:, 2] = (
            matrix[:, 0] * (center ** 2 - width ** 2 - 2 * center * axis + axis ** 2) / width ** 4
        )
    return matrix
