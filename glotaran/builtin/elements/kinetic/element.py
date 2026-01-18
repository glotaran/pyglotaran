from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.builtin.elements.kinetic.kinetic import Kinetic
from glotaran.builtin.elements.kinetic.matrix import calculate_matrix_gaussian_activation
from glotaran.builtin.elements.kinetic.matrix import calculate_matrix_gaussian_activation_on_index
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model.element import ExtendableElement
from glotaran.model.item import ParameterType  # noqa:F401

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class KineticElement(ExtendableElement, Kinetic):
    type: Literal["kinetic"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "kinetic"
    data_model_type: ClassVar[type[DataModel]] = ActivationDataModel  # type:ignore[valid-type]
    dimension: str = "time"

    def extend(self, other: KineticElement) -> KineticElement:  # type:ignore[override]
        return other.model_copy(update={"rates": self.rates | other.rates})

    # TODO: consolidate parent method.
    @classmethod
    def combine(cls, kinetics: list[KineticElement]) -> KineticElement:  # type:ignore[override]
        """Create a combined matrix.

        When combining k-matrices km1 and km2 (km1.combine(km2)),
        entries in km1 will be overwritten by corresponding entries in km2.

        Parameters
        ----------
        k_matrix :
            KMatrix to combine with.

        Returns
        -------
        combined :
            The combined KMatrix.
        """
        return cls(
            type="kinetic",
            rates=reduce(lambda lhs, rhs: lhs | rhs, [k.rates for k in kinetics]),
            label="",
        )

    @staticmethod
    def combine_matrices(lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
        if lhs.shape != rhs.shape:
            if len(lhs.shape) > len(rhs):
                return lhs + rhs[np.newaxis, :, :]
            return lhs[np.newaxis, :, :] + rhs
        return lhs + rhs

    def calculate_matrix(  # type:ignore[override]
        self,
        model: ActivationDataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[list[str], ArrayLike]:
        compartments = self.compartments
        matrices = []
        for activation in model.activations.values():
            initial_concentrations = np.array(
                [float(activation.compartments.get(label, 0)) for label in compartments]
            )
            normalized_compartments = [
                c not in activation.not_normalized_compartments for c in compartments
            ]
            initial_concentrations[normalized_compartments] /= np.sum(
                initial_concentrations[normalized_compartments]
            )
            rates = self.calculate(initial_concentrations)

            matrix = (
                self.calculate_matrix_gaussian_activation(
                    activation, global_axis, model_axis, compartments, rates
                )
                if isinstance(activation, MultiGaussianActivation)
                else np.exp(np.outer(model_axis, -rates))
            )

            if not np.all(np.isfinite(matrix)):
                msg = f"Non-finite concentrations for kinetic of element '{self.label}'"
                raise ValueError(msg)

            # apply A matrix
            matrix = matrix @ self.a_matrix(initial_concentrations)
            matrices.append(matrix)

        return compartments, matrices[0] if len(matrices) == 1 else reduce(
            self.combine_matrices, matrices
        )

    def calculate_matrix_gaussian_activation(
        self,
        activation: MultiGaussianActivation,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        compartments: list[str],
        rates: ArrayLike,
    ) -> ArrayLike:
        parameters = activation.parameters(global_axis)
        matrix_shape = (model_axis.size, len(compartments))
        index_dependent = any(isinstance(p, list) for p in parameters)
        if index_dependent:
            matrix_shape = (global_axis.size, *matrix_shape)  # type:ignore[assignment]
        matrix = np.zeros(matrix_shape, dtype=np.float64)
        scales = np.array([p.scale for p in (parameters[0] if index_dependent else parameters)])  # type:ignore[union-attr]
        if index_dependent:
            calculate_matrix_gaussian_activation(
                matrix,
                rates,
                model_axis,
                np.array([[p.center for p in ps] for ps in parameters]),
                np.array([[p.width for p in ps] for ps in parameters]),
                scales,
                parameters[0][0].backsweep_period,
            )
        else:
            calculate_matrix_gaussian_activation_on_index(
                matrix,
                rates,
                model_axis,
                np.array([p.center for p in parameters]),  # type:ignore[attr-defined]
                np.array([p.width for p in parameters]),  # type:ignore[attr-defined]
                scales,
                parameters[0].backsweep_period,  # type:ignore[attr-defined]
            )
        if activation.normalize:
            matrix /= np.sum(scales)

        return matrix

    def create_result(
        self,
        model: ActivationDataModel,  # type:ignore[override]
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        species_amplitudes = amplitudes.sel(amplitude_label=self.compartments).rename(
            amplitude_label="compartment"
        )
        species_concentrations = concentrations.sel(amplitude_label=self.compartments).rename(
            amplitude_label="compartment"
        )

        k_matrix = xr.DataArray(
            self.full_array, coords={"to": self.compartments, "from": self.compartments}
        )
        reduced_k_matrix = xr.DataArray(
            self.array, coords={"to": self.compartments, "from": self.compartments}
        )

        # TODO: do we want to store it in this format?
        rates = self.calculate()
        lifetimes = 1 / rates
        kinetic_coords = {
            "kinetic": np.arange(1, rates.size + 1),
            "rate": ("kinetic", rates),
            "lifetime": ("kinetic", lifetimes),
        }

        initial_concentrations_list = []
        a_matrices = []
        kinetic_amplitudes_list = []
        activation_names = list(model.activations.keys())
        for activation in model.activations.values():
            initial_concentration_activation = np.array(
                [float(activation.compartments.get(label, 0)) for label in self.compartments]
            )
            initial_concentrations_list.append(initial_concentration_activation)
            a_matrix = self.a_matrix(initial_concentration_activation)
            a_matrices.append(a_matrix)
            kinetic_amplitudes_list.append(species_amplitudes.to_numpy() @ a_matrix.T)

        initial_concentrations = xr.DataArray(
            initial_concentrations_list,
            coords={
                "activation": activation_names,
                "compartment": self.compartments,
            },
        )
        a_matrix = xr.DataArray(
            a_matrices,
            coords={
                "activation": activation_names,
                "compartment": self.compartments,
            }
            | kinetic_coords,
            dims=("activation", "compartment", "kinetic"),
        )
        kinetic_amplitudes_coords = (
            {"activation": activation_names} | kinetic_coords | dict(species_amplitudes.coords)
        )
        del kinetic_amplitudes_coords["compartment"]
        kinetic_amplitudes = xr.DataArray(
            kinetic_amplitudes_list,
            coords=kinetic_amplitudes_coords,
            dims=("activation", global_dimension, "kinetic"),
        )

        return xr.Dataset(
            {
                "amplitudes": species_amplitudes,
                "concentrations": species_concentrations,
                "initial_concentrations": initial_concentrations,
                "kinetic_amplitudes": kinetic_amplitudes,
                "k_matrix": k_matrix,
                "reduced_k_matrix": reduced_k_matrix,
                "a_matrix": a_matrix,
            }
        )
