from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.builtin.items.activation import add_activation_to_result_data
from glotaran.model.data_model import DataModel  # noqa: TCH001
from glotaran.model.element import Element
from glotaran.model.errors import GlotaranModelError
from glotaran.model.item import ParameterType  # noqa: TCH001

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class CoherentArtifactElement(Element):
    type: Literal["coherent-artifact"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "coherent-artifact"
    dimension: str = "time"
    data_model_type: ClassVar[type[DataModel]] = ActivationDataModel  # type:ignore[valid-type]
    order: int
    width: ParameterType | None = None

    def calculate_matrix(  # type:ignore[override]
        self,
        model: ActivationDataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
    ):
        if not 1 <= self.order <= 3:
            raise GlotaranModelError("Coherent artifact order must be between in [1,3]")

        activations = [a for a in model.activation if isinstance(a, MultiGaussianActivation)]

        matrices = []
        activation_indices = []
        for i, activation in enumerate(activations):
            if self.label not in activation.compartments:
                continue
            activation_indices.append(i)
            parameters = activation.parameters(global_axis)

            matrix_shape = (model_axis.size, self.order)
            index_dependent = any(isinstance(p, list) for p in parameters)
            if index_dependent:
                matrix_shape = (global_axis.size, *matrix_shape)  # type:ignore[assignment]
            matrix = np.zeros(matrix_shape, dtype=np.float64)
            if index_dependent:
                _calculate_coherent_artifact_matrix(
                    matrix,
                    np.array([ps[0].center for ps in parameters]),  # type:ignore[index]
                    np.array(
                        [self.width or ps[0].width for ps in parameters]  # type:ignore[index]
                    ),
                    global_axis.size,
                    model_axis,
                    self.order,
                )

            else:
                _calculate_coherent_artifact_matrix_on_index(
                    matrix,
                    parameters[0].center,  # type:ignore[union-attr]
                    self.width or parameters[0].width,  # type:ignore[union-attr]
                    model_axis,
                    self.order,
                )
            matrix *= activation.compartments[self.label]  # type:ignore[arg-type]
            matrices.append(matrix)

        if not len(matrices):
            raise GlotaranModelError(
                f'No (multi-)gaussian activation for coherent-artifact "{self.label}".'
            )

        clp_axis = []
        for i in activation_indices:
            clp_axis += [f"{label}_activation_{i}" for label in self.compartments()]
        return clp_axis, np.concatenate(matrices, axis=len(matrices[0].shape) - 1)

    def compartments(self):
        return [f"coherent_artifact_{self.label}_order_{i}" for i in range(1, self.order + 1)]

    def add_to_result_data(  # type:ignore[override]
        self,
        model: ActivationDataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        add_activation_to_result_data(model, data)
        if "coherent_artifact_order" in data.coords:
            return

        data_matrix = data.global_matrix if "global_matrix" in data else data.matrix
        elements = [m for m in model.elements if isinstance(m, CoherentArtifactElement)]
        nr_activations = data.gaussian_activation.size
        matrices = []
        estimations = []
        for coherent_artifact in elements:
            artifact_matrices = []
            artifact_estimations = []
            activation_indices = []
            for i in range(nr_activations):
                clp_axis = [
                    label
                    for label in data.clp_label.data
                    if label.startswith(f"coherent_artifact_{coherent_artifact.label}")
                    and label.endswith(f"_activation_{i}")
                ]
                if not len(clp_axis):
                    continue
                activation_indices.append(i)
                order = [label.split("_activation_")[0].split("_order")[1] for label in clp_axis]

                artifact_matrices.append(
                    data_matrix.sel(clp_label=clp_axis)
                    .rename(clp_label="coherent_artifact_order")
                    .assign_coords({"coherent_artifact_order": order})
                )
                if "global_matrix" not in data:
                    artifact_estimations.append(
                        data.clp.sel(clp_label=clp_axis)
                        .rename(clp_label="coherent_artifact_order")
                        .assign_coords({"coherent_artifact_order": order})
                    )
            matrices.append(
                xr.concat(artifact_matrices, dim="gaussian_activation").assign_coords(
                    {"gaussian_activation": activation_indices}
                )
            )
            if "global_matrix" not in data:
                estimations.append(
                    xr.concat(artifact_estimations, dim="gaussian_activation").assign_coords(
                        {"gaussian_activation": activation_indices}
                    )
                )
        data["coherent_artifact_response"] = xr.concat(
            matrices, dim="coherent_artifact"
        ).assign_coords({"coherent_artifact": [m.label for m in elements]})
        if "global_matrix" not in data:
            data["coherent_artifact_associated_estimation"] = xr.concat(
                estimations, dim="coherent_artifact"
            ).assign_coords({"coherent_artifact": [m.label for m in elements]})


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
