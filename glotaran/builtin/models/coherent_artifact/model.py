from __future__ import annotations

from typing import Literal

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model import GlotaranModelError
from glotaran.model import Model
from glotaran.model import ParameterType


class CoherentArtifactModel(Model):
    type: Literal["coherent-artifact"]
    register_as = "coherent-artifact"
    dimension = "time"
    unique = True
    data_model = ActivationDataModel
    order: int
    width: ParameterType | None = None

    def calculate_matrix(
        self,
        model: ActivationDataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ):
        if not 1 <= self.order <= 3:
            raise GlotaranModelError("Coherent artifact order must be between in [1,3]")

        activations = [a for a in model.activation if isinstance(a, MultiGaussianActivation)]
        if not len(activations):
            raise GlotaranModelError(
                f'No (multi-)gaussian activation in dataset with coherent-artifact "{self.label}".'
            )

        matrices = []
        for activation in activations:
            parameters = activation.parameters(global_axis)

            matrix_shape = (model_axis.size, self.order)
            index_dependent = any(isinstance(p, list) for p in parameters)
            if index_dependent:
                matrix_shape = (global_axis.size,) + matrix_shape
            matrix = np.zeros(matrix_shape, dtype=np.float64)
            if index_dependent:
                _calculate_coherent_artifact_matrix(
                    matrix,
                    np.array([ps[0].center for ps in parameters]),
                    np.array([self.width or ps[0].width for ps in parameters]),
                    global_axis.size,
                    model_axis,
                    self.order,
                )

            else:
                _calculate_coherent_artifact_matrix_on_index(
                    matrix,
                    parameters[0].center,
                    self.width or parameters[0].width,
                    model_axis,
                    self.order,
                )
            matrices.append(matrix)

        if len(matrices) == 1:
            return self.compartments(), matrices[0]

        clp_axis = []
        for i in range(len(matrices)):
            clp_axis += [f"{label}_activation_{i}" for label in self.compartments()]
        return clp_axis, np.concatenate(matrices, axis=len(matrices[0].shape) - 1)

    def compartments(self):
        return [f"coherent_artifact_order_{i}" for i in range(1, self.order + 1)]

    def add_to_result_data(
        self,
        model: ActivationDataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        if not as_global:
            nr_activations = len(
                [a for a in model.activation if isinstance(a, MultiGaussianActivation)]
            )
            global_dimension = data.attrs["global_dimension"]
            model_dimension = data.attrs["model_dimension"]
            data.coords["coherent_artifact_order"] = np.arange(1, self.order + 1)
            response_dimensions = (model_dimension, "coherent_artifact_order")
            if len(data.matrix.shape) == 3:
                response_dimensions = (global_dimension, *response_dimensions)
            if nr_activations == 1:
                data["coherent_artifact_response"] = (
                    response_dimensions,
                    data.matrix.sel(clp_label=self.compartments()).values,
                )
                data["coherent_artifact_associated_estimation"] = (
                    (global_dimension, "coherent_artifact_order"),
                    data.clp.sel(clp_label=self.compartments()).values,
                )
            else:
                for i in range(nr_activations):
                    clp_axis = [f"{label}_activation_{i}" for label in self.compartments()]
                    data["coherent_artifact_response_activation_{i}"] = (
                        response_dimensions,
                        data.matrix.sel(clp_label=clp_axis).values,
                    )
                    data["coherent_artifact_associated_estimation_activation_{i}"] = (
                        (global_dimension, "coherent_artifact_order"),
                        data.clp.sel(clp_label=clp_axis).values,
                    )


@nb.jit(nopython=True, parallel=False)
def _calculate_coherent_artifact_matrix(
    matrix, centers, widths, global_axis_size, model_axis, order
):
    for i in nb.prange(global_axis_size):
        _calculate_coherent_artifact_matrix_on_index(
            matrix[i], centers[i], widths[i], model_axis, order
        )


@nb.jit(nopython=True, parallel=True)
def _calculate_coherent_artifact_matrix_on_index(
    matrix: np.ndarray, center: float, width: float, axis: np.ndarray, order: int
):

    matrix[:, 0] = np.exp(-1 * (axis - center) ** 2 / (2 * width**2))
    if order > 1:
        matrix[:, 1] = matrix[:, 0] * (center - axis) / width**2

    if order > 2:
        matrix[:, 2] = (
            matrix[:, 0] * (center**2 - width**2 - 2 * center * axis + axis**2) / width**4
        )
