"""This package contains the kinetic megacomplex item."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.decay_parallel_megacomplex import DecayDatasetModel
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.builtin.megacomplexes.decay.util import index_dependent
from glotaran.builtin.megacomplexes.decay.util import retrieve_irf
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import ParameterType
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@megacomplex(dataset_model_type=DecayDatasetModel, unique=True)
class CoherentArtifactMegacomplex(Megacomplex):
    dimension: str = "time"
    type: str = "coherent-artifact"
    order: int
    width: ParameterType | None = None

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        if not 1 <= self.order <= 3:
            raise ModelError("Coherent artifact order must be between in [1,3]")

        irf = dataset_model.irf
        if irf is None:
            raise ModelError(f'No irf in dataset "{dataset_model.label}"')

        if not isinstance(irf, IrfMultiGaussian):
            raise ModelError(f'Irf in dataset "{dataset_model.label} is not a gaussian irf."')

        matrix_shape = (
            (global_axis.size, model_axis.size, self.order)
            if index_dependent(dataset_model)
            else (model_axis.size, self.order)
        )
        matrix = np.zeros(matrix_shape, dtype=np.float64)
        if index_dependent(dataset_model):
            centers, widths = [], []
            for global_index in range(global_axis.size):
                center, width = self.get_irf_parameter(irf, global_index, global_axis)
                centers.append(center)
                widths.append(width)
            _calculate_coherent_artifact_matrix(
                matrix,
                np.asarray(centers),
                np.asarray(widths),
                global_axis.size,
                model_axis,
                self.order,
            )

        else:
            center, width = self.get_irf_parameter(irf, None, global_axis)
            _calculate_coherent_artifact_matrix_on_index(
                matrix, center, width, model_axis, self.order
            )

        return self.compartments(), matrix

    def get_irf_parameter(
        self, irf: IrfMultiGaussian, global_index: int | None, global_axis: ArrayLike
    ) -> tuple[float, float]:
        center, width, _, shift, _, _ = irf.parameter(global_index, global_axis)
        center = center[0] - shift
        width = self.width.value if self.width is not None else width[0]
        return center, width

    def compartments(self):
        return [f"coherent_artifact_{i}_{self.label}" for i in range(1, self.order + 1)]

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        global_dimension = dataset.attrs["global_dimension"]
        if not is_full_model:
            model_dimension = dataset.attrs["model_dimension"]
            dataset.coords["coherent_artifact_order"] = np.arange(1, self.order + 1)
            response_dimensions = (model_dimension, "coherent_artifact_order")
            if len(dataset.matrix.shape) == 3:
                response_dimensions = (global_dimension, *response_dimensions)
            dataset["coherent_artifact_response"] = (
                response_dimensions,
                dataset.matrix.sel(clp_label=self.compartments()).values,
            )
            dataset["coherent_artifact_associated_spectra"] = (
                (global_dimension, "coherent_artifact_order"),
                dataset.clp.sel(clp_label=self.compartments()).values,
            )
        retrieve_irf(dataset_model, dataset, global_dimension)


@nb.jit(nopython=True, parallel=False)
def _calculate_coherent_artifact_matrix(
    matrix, centers, widths, global_axis_size, model_axis, order
):
    for i in nb.prange(global_axis_size):
        _calculate_coherent_artifact_matrix_on_index(
            matrix[i], centers[i], widths[i], model_axis, order
        )


@nb.jit(nopython=True, parallel=True)
def _calculate_coherent_artifact_matrix_on_index(matrix, center, width, axis, order):
    matrix[:, 0] = np.exp(-1 * (axis - center) ** 2 / (2 * width**2))
    if order > 1:
        matrix[:, 1] = matrix[:, 0] * (center - axis) / width**2

    if order > 2:
        matrix[:, 2] = (
            matrix[:, 0] * (center**2 - width**2 - 2 * center * axis + axis**2) / width**4
        )
