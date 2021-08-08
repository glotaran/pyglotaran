"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.initial_concentration import InitialConcentration
from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import decay_matrix_implementation
from glotaran.builtin.megacomplexes.decay.util import retrieve_decay_associated_data
from glotaran.builtin.megacomplexes.decay.util import retrieve_irf
from glotaran.builtin.megacomplexes.decay.util import retrieve_species_associated_data
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex


@megacomplex(
    dimension="time",
    model_items={
        "k_matrix": List[KMatrix],
    },
    properties={},
    dataset_model_items={
        "initial_concentration": {"type": InitialConcentration, "allow_none": True},
        "irf": {"type": Irf, "allow_none": True},
    },
    register_as="decay",
)
class DecayMegacomplex(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""

    def has_k_matrix(self) -> bool:
        return len(self.k_matrix) != 0

    def full_k_matrix(self, model=None):
        full_k_matrix = None
        for k_matrix in self.k_matrix:
            if model:
                k_matrix = model.k_matrix[k_matrix]
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        return full_k_matrix

    @property
    def involved_compartments(self):
        return self.full_k_matrix().involved_compartments() if self.full_k_matrix() else []

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return (
            isinstance(dataset_model.irf, IrfMultiGaussian)
            and dataset_model.irf.is_index_dependent()
        )

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):
        if dataset_model.initial_concentration is None:
            raise ModelError(
                f'No initial concentration specified in dataset "{dataset_model.label}"'
            )
        initial_concentration = dataset_model.initial_concentration.normalized()

        k_matrix = self.full_k_matrix()

        # we might have more species in the model then in the k matrix
        species = [
            comp
            for comp in initial_concentration.compartments
            if comp in k_matrix.involved_compartments()
        ]

        # the rates are the eigenvalues of the k matrix
        rates = k_matrix.rates(initial_concentration)

        global_dimension = dataset_model.get_global_dimension()
        global_index = indices.get(global_dimension)
        global_axis = dataset_model.get_global_axis()
        model_axis = dataset_model.get_model_axis()

        # init the matrix
        size = (model_axis.size, rates.size)
        matrix = np.zeros(size, dtype=np.float64)

        decay_matrix_implementation(
            matrix, rates, global_index, global_axis, model_axis, dataset_model
        )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"Non-finite concentrations for K-Matrix '{k_matrix.label}':\n"
                f"{k_matrix.matrix_as_markdown(fill_parameters=True)}"
            )

        # apply A matrix
        matrix = matrix @ k_matrix.a_matrix(initial_concentration)

        # done
        return species, matrix

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        global_dimension = dataset_model.get_global_dimension()
        name = "images" if global_dimension == "pixel" else "spectra"

        species_dimension = "decay_species" if as_global else "species"
        if species_dimension not in dataset.coords:
            # We are the first Decay complex called and add SAD for all decay megacomplexes
            retrieve_species_associated_data(
                dataset_model,
                dataset,
                species_dimension,
                global_dimension,
                name,
                is_full_model,
                as_global,
            )
        if isinstance(dataset_model.irf, IrfMultiGaussian) and "irf" not in dataset:
            retrieve_irf(dataset_model, dataset, global_dimension)

        if not is_full_model:
            multiple_complexes = (
                len([m for m in dataset_model.megacomplex if isinstance(m, DecayMegacomplex)]) > 1
            )
            retrieve_decay_associated_data(
                self,
                dataset_model,
                dataset,
                global_dimension,
                name,
                multiple_complexes,
            )
