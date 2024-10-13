from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class KineticElement(ExtendableElement, Kinetic):
    type: Literal["kinetic"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "kinetic"
    data_model_type: ClassVar[type[DataModel]] = ActivationDataModel  # type:ignore[valid-type]
    dimension: str = "time"

    def extend(self, other: KineticElement):  # type:ignore[override]
        return other.model_copy(update={"rates": self.rates | other.rates})

    # TODO: consolidate parent method.
    @classmethod
    def combine(cls, kinetics: list[KineticElement]) -> KineticElement:  # type:ignore[override]
        """Creates a combined matrix.

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
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        compartments = self.compartments
        matrices = []
        for _, activation in model.activations.items():
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
                raise ValueError(
                    f"Non-finite concentrations for kinetic of element '{self.label}'"
                )

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
        scales = np.array(
            [
                p.scale  # type:ignore[union-attr]
                for p in (
                    parameters[0] if index_dependent else parameters  # type:ignore[union-attr]
                )
            ]
        )
        if index_dependent:
            calculate_matrix_gaussian_activation(
                matrix,
                rates,
                model_axis,
                np.array([[p.center for p in ps] for ps in parameters]),  # type:ignore[union-attr]
                np.array([[p.width for p in ps] for ps in parameters]),  # type:ignore[union-attr]
                scales,
                parameters[0][0].backsweep_period,  # type:ignore[index]
            )
        else:
            calculate_matrix_gaussian_activation_on_index(
                matrix,
                rates,
                model_axis,
                np.array([p.center for p in parameters]),  # type:ignore[union-attr]
                np.array([p.width for p in parameters]),  # type:ignore[union-attr]
                scales,
                parameters[0].backsweep_period,  # type:ignore[union-attr]
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
        species_amplitude = amplitudes.sel(amplitude_label=self.compartments).rename(
            amplitude_label="compartment"
        )
        species_concentration = concentrations.sel(amplitude_label=self.compartments).rename(
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

        initial_concentrations = []
        a_matrices = []
        kinetic_amplitudes = []
        for _, activation in model.activations.items():
            initial_concentration = np.array(
                [float(activation.compartments.get(label, 0)) for label in self.compartments]
            )
            initial_concentrations.append(initial_concentration)
            a_matrix = self.a_matrix(initial_concentration)
            a_matrices.append(a_matrix)
            kinetic_amplitudes.append(species_amplitude.to_numpy() @ a_matrix.T)

        initial_concentration = xr.DataArray(
            initial_concentrations,
            coords={
                "activation": range(len(initial_concentrations)),
                "compartment": self.compartments,
            },
        )
        a_matrix = xr.DataArray(
            a_matrices,
            coords={
                "activation": range(len(a_matrices)),
                "compartment": self.compartments,
            }
            | kinetic_coords,
            dims=("activation", "compartment", "kinetic"),
        )
        kinetic_amplitude_coords = (
            {"activation": range(len(kinetic_amplitudes))}
            | kinetic_coords
            | dict(species_amplitude.coords)
        )
        del kinetic_amplitude_coords["compartment"]
        kinetic_amplitude = xr.DataArray(
            kinetic_amplitudes,
            coords=kinetic_amplitude_coords,
            dims=("activation", global_dimension, "kinetic"),
        )

        return xr.Dataset(
            {
                "amplitudes": species_amplitude,
                "concentrations": species_concentration,
                "initial_concentration": initial_concentration,
                "kinetic_amplitude": kinetic_amplitude,
                "k_matrix": k_matrix,
                "reduced_k_matrix": reduced_k_matrix,
                "a_matrix": a_matrix,
            }
        )
