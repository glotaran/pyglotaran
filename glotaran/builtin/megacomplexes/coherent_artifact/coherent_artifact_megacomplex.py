"""This package contains the kinetic megacomplex item."""
from __future__ import annotations

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex
from glotaran.parameter import Parameter


@megacomplex(
    dimension="time",
    unique=True,
    properties={
        "order": {"type": int},
        "width": {"type": Parameter, "allow_none": True},
    },
    dataset_model_items={
        "irf": {"type": Irf, "allow_none": True},
    },
    register_as="coherent-artifact",
)
class CoherentArtifactMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):
        if not 1 <= self.order <= 3:
            raise ModelError("Coherent artifact order must be between in [1,3]")

        if dataset_model.irf is None:
            raise ModelError(f'No irf in dataset "{dataset_model.label}"')

        if not isinstance(dataset_model.irf, IrfMultiGaussian):
            raise ModelError(f'Irf in dataset "{dataset_model.label} is not a gaussian irf."')

        global_dimension = dataset_model.get_global_dimension()
        global_index = indices.get(global_dimension)
        global_axis = dataset_model.get_global_axis()
        model_axis = dataset_model.get_model_axis()

        irf = dataset_model.irf

        center, width, _, _, _, _ = irf.parameter(global_index, global_axis)
        center = center[0]
        width = self.width.value if self.width is not None else width[0]

        matrix = _calculate_coherent_artifact_matrix(center, width, model_axis, self.order)
        return self.compartments(), matrix

    def compartments(self):
        return [f"coherent_artifact_{i}" for i in range(1, self.order + 1)]

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return False

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        if not is_full_model:
            global_dimension = dataset_model.get_global_dimension()
            model_dimension = dataset_model.get_model_dimension()
            dataset.coords["coherent_artifact_order"] = np.arange(1, self.order + 1)
            response_dimensions = (model_dimension, "coherent_artifact_order")
            if dataset_model.is_index_dependent() is True:
                response_dimensions = (global_dimension, *response_dimensions)
            dataset["coherent_artifact_response"] = (
                response_dimensions,
                dataset.matrix.sel(clp_label=self.compartments()).values,
            )
            dataset["coherent_artifact_associated_spectra"] = (
                (global_dimension, "coherent_artifact_order"),
                dataset.clp.sel(clp_label=self.compartments()).values,
            )


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
