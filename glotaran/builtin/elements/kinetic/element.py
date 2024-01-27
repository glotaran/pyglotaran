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
from glotaran.builtin.items.activation import add_activation_to_result_data
from glotaran.model.data_model import DataModel
from glotaran.model.data_model import is_data_model_global
from glotaran.model.element import ExtendableElement

if TYPE_CHECKING:
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
        compartments = self.species
        matrices = []
        for activation in model.activation:
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

    def add_to_result_data(  # type:ignore[override]
        self, model: ActivationDataModel, data: xr.Dataset, as_global: bool
    ):
        add_activation_to_result_data(model, data)
        if "species" in data.coords or is_data_model_global(model):
            return
        kinetic = self.combine([m for m in model.elements if isinstance(m, KineticElement)])
        species = kinetic.species
        global_dimension = data.attrs["global_dimension"]
        model_dimension = data.attrs["model_dimension"]

        data.coords["species"] = species
        matrix = data.global_matrix if as_global else data.matrix
        clp_dim = "global_clp_label" if as_global else "clp_label"
        concentration_shape = (
            global_dimension if as_global else model_dimension,
            "species",
        )
        if len(matrix.shape) > 2:
            concentration_shape = (  # type:ignore[assignment]
                (model_dimension if as_global else global_dimension),
                *concentration_shape,
            )
        data["species_concentration"] = (
            concentration_shape,
            matrix.sel({clp_dim: species}).to_numpy(),
        )

        data["k_matrix"] = xr.DataArray(kinetic.full_array, dims=(("species"), ("species")))
        data["k_matrix_reduced"] = xr.DataArray(kinetic.array, dims=(("species"), ("species")))

        rates = kinetic.calculate()
        lifetimes = 1 / rates
        data.coords["kinetic"] = np.arange(1, rates.size + 1)
        data.coords["rate"] = ("kinetic", rates)
        data.coords["lifetime"] = ("kinetic", lifetimes)

        if hasattr(data, "global_matrix"):
            return

        species_associated_estimation = data.clp.sel(clp_label=species).data
        data["species_associated_estimation"] = (
            (global_dimension, "species"),
            species_associated_estimation,
        )
        initial_concentrations = []
        a_matrices = []
        kinetic_associated_estimations = []
        for activation in model.activation:
            initial_concentration = np.array(
                [float(activation.compartments.get(label, 0)) for label in species]
            )
            initial_concentrations.append(initial_concentration)
            a_matrix = kinetic.a_matrix(initial_concentration)
            a_matrices.append(a_matrix)
            kinetic_associated_estimations.append(species_associated_estimation @ a_matrix.T)

        data["initial_concentration"] = (
            ("activation", "species"),
            initial_concentrations,
        )
        data["a_matrix"] = (
            ("activation", "species", "kinetic"),
            a_matrices,
        )
        data["kinetic_associated_estimation"] = (
            ("activation", global_dimension, "kinetic"),
            kinetic_associated_estimations,
        )
