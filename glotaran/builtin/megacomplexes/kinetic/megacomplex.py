from functools import reduce
from typing import Literal

import numpy as np

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.builtin.items.kinetic import Kinetic
from glotaran.builtin.megacomplexes.kinetic.matrix import calculate_matrix_gaussian_activation
from glotaran.builtin.megacomplexes.kinetic.matrix import (
    calculate_matrix_gaussian_activation_on_index,
)
from glotaran.model import LibraryItemType
from glotaran.model import Megacomplex


class KineticMegacomplex(Megacomplex):
    type: Literal["kinetic"]
    register_as = "kinetic"
    data_model = ActivationDataModel
    dimension: str = "time"
    kinetic: list[LibraryItemType[Kinetic]]

    @staticmethod
    def reduce_matrices(lhs: np.typing.ArrayLike, rhs: np.typing.ArrayLike) -> np.typing.ArrayLike:
        if lhs.shape != rhs.shape:
            if len(lhs.shape) > len(rhs):
                return lhs + rhs[np.newaxis, :, :]
            else:
                return lhs[np.newaxis, :, :] + rhs
        return lhs + rhs

    def calculate_matrix(
        self,
        model: ActivationDataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ) -> tuple[list[str], np.typing.ArrayLike]:
        kinetic = Kinetic.combine(self.kinetic)
        compartments = kinetic.compartments
        matrices = []
        for activation in model.activation:
            initial_concentrations = np.array(
                [activation.compartments.get(label, 0) for label in compartments]
            )
            normalized_compartments = [
                c not in activation.not_normalized_compartments for c in compartments
            ]
            initial_concentrations[normalized_compartments] /= np.sum(
                initial_concentrations[normalized_compartments]
            )
            rates = kinetic.rates(initial_concentrations)

            matrix = (
                self.calculate_matrix_gaussian_activation(
                    activation, global_axis, model_axis, compartments, rates
                )
                if isinstance(activation, MultiGaussianActivation)
                else np.exp(np.outer(model_axis, -rates))
            )

            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"Non-finite concentrations for kinetic of data model '{model.label}':\n"
                    f"{kinetic.matrix_as_markdown()}"
                )

            # apply A matrix
            matrix = matrix @ kinetic.a_matrix(initial_concentrations)
            matrices.append(matrix)

        return compartments, matrices[0] if len(matrices) == 1 else reduce(
            self.reduce_matrices, matrices
        )

    def calculate_matrix_gaussian_activation(
        self,
        activation: MultiGaussianActivation,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        compartments: list[str],
        rates: np.typing.ArrayLike,
    ) -> np.typing.ndarray:
        parameters = activation.parameters(global_axis)
        matrix_shape = (model_axis.size, len(compartments))
        index_dependent = any(isinstance(p, list) for p in parameters)
        if index_dependent:
            matrix_shape = (global_axis.size,) + matrix_shape
        matrix = np.zeros(matrix_shape, dtype=np.float64)
        scales = [p.scale for p in (parameters[0] if index_dependent else parameters)]
        if index_dependent:
            calculate_matrix_gaussian_activation(
                matrix,
                rates,
                model_axis,
                [[p.center for p in ps] for ps in parameters],
                [[p.width for p in ps] for ps in parameters],
                scales,
                parameters[0][0].backsweep,
                parameters[0][0].backsweep_period,
            )
        else:
            calculate_matrix_gaussian_activation_on_index(
                matrix,
                rates,
                model_axis,
                [p.center for p in parameters],
                [p.width for p in parameters],
                scales,
                parameters[0].backsweep,
                parameters[0].backsweep_period,
            )
        if activation.normalize:
            matrix /= np.sum(scales)

        return matrix
