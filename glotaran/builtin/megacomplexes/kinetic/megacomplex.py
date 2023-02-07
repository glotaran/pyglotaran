from functools import reduce
from typing import Literal

import numpy as np
import xarray as xr

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
    data_model_type = ActivationDataModel
    dimension: str = "time"
    kinetic: list[LibraryItemType[Kinetic]]

    @staticmethod
    def combine_matrices(
        lhs: np.typing.ArrayLike, rhs: np.typing.ArrayLike
    ) -> np.typing.ArrayLike:
        if lhs.shape != rhs.shape:
            if len(lhs.shape) > len(rhs):
                return lhs + rhs[np.newaxis, :, :]
            else:
                return lhs[np.newaxis, :, :] + rhs
        return lhs + rhs

    def get_species(self) -> list[str]:
        return Kinetic.combine(self.kinetic).compartments

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
                [float(activation.compartments.get(label, 0)) for label in compartments]
            )
            normalized_compartments = [
                c not in activation.not_normalized_compartments for c in compartments
            ]
            initial_concentrations[normalized_compartments] /= np.sum(
                initial_concentrations[normalized_compartments]
            )
            rates = kinetic.calculate(initial_concentrations)

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
            self.combine_matrices, matrices
        )

    def calculate_matrix_gaussian_activation(
        self,
        activation: MultiGaussianActivation,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        compartments: list[str],
        rates: np.typing.ArrayLike,
    ) -> np.typing.ArrayLike:
        parameters = activation.parameters(global_axis)
        matrix_shape = (model_axis.size, len(compartments))
        index_dependent = any(isinstance(p, list) for p in parameters)
        if index_dependent:
            matrix_shape = (global_axis.size,) + matrix_shape
        matrix = np.zeros(matrix_shape, dtype=np.float64)
        scales = np.array([p.scale for p in (parameters[0] if index_dependent else parameters)])
        if index_dependent:
            calculate_matrix_gaussian_activation(
                matrix,
                rates,
                model_axis,
                np.array([[p.center for p in ps] for ps in parameters]),
                np.array([[p.width for p in ps] for ps in parameters]),
                scales,
                parameters[0][0].backsweep,
                parameters[0][0].backsweep_period,
            )
        else:
            calculate_matrix_gaussian_activation_on_index(
                matrix,
                rates,
                model_axis,
                np.array([p.center for p in parameters]),
                np.array([p.width for p in parameters]),
                scales,
                parameters[0].backsweep,
                parameters[0].backsweep_period,
            )
        if activation.normalize:
            matrix /= np.sum(scales)

        return matrix

    def add_to_result_data(self, model: ActivationDataModel, data: xr.Dataset, as_global: bool):
        MultiGaussianActivation.add_to_result_data(model, data)
        if "species" in data.coords:
            return
        megacomplexes = [m for m in model.megacomplex if isinstance(m, KineticMegacomplex)]
        kinetic = Kinetic.combine([k for m in megacomplexes for k in m.kinetic])
        species = kinetic.compartments
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
            concentration_shape = (
                (model_dimension if as_global else global_dimension),
            ) + concentration_shape
        data["species_concentration"] = (
            concentration_shape,
            matrix.sel({clp_dim: species}).values,
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
            kinetic_associated_estimations,
        )
        data["kinetic_associated_estimation"] = (
            ("activation", global_dimension, "kinetic"),
            kinetic_associated_estimations,
        )
