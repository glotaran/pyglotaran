from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model.element import Element
from glotaran.model.errors import GlotaranModelError
from glotaran.model.item import ParameterType  # noqa: TCH001

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
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

        activations = [
            a
            for a in model.activation
            if isinstance(a, MultiGaussianActivation) and self.label in a.compartments
        ]

        if not len(activations):
            raise GlotaranModelError(
                f'No (multi-)gaussian activation for coherent-artifact "{self.label}".'
            )
        if len(activations) > 1:
            raise GlotaranModelError(
                f'Coherent artifact "{self.label}" must be associated with exactly one activation.'
            )
        activation = activations[0]

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

        return self.compartments, matrix

    @property
    def compartments(self):
        return [f"coherent_artifact_{self.label}_order_{i}" for i in range(1, self.order + 1)]

    def create_result(
        self,
        model: ActivationDataModel,  # type:ignore[override]
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        amplitude = (
            amplitudes.sel(amplitude_label=self.compartments)
            .rename(amplitude_label="coherent_artifact_order")
            .assign_coords({"coherent_artifact_order": range(1, self.order + 1)})
        )
        concentration = (
            concentrations.sel(amplitude_label=self.compartments)
            .rename(amplitude_label="coherent_artifact_order")
            .assign_coords({"coherent_artifact_order": range(1, self.order + 1)})
        )
        return xr.Dataset({"amplitudes": amplitude, "concentrations": concentration})


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
