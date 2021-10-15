"""This package contains the decay megacomplex item."""
from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import decay_matrix_implementation
from glotaran.builtin.megacomplexes.decay.util import retrieve_decay_associated_data
from glotaran.builtin.megacomplexes.decay.util import retrieve_irf
from glotaran.builtin.megacomplexes.decay.util import retrieve_species_associated_data
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex


class DecayMegacomplexBase(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""

    def get_compartments(self, dataset_model: DatasetModel) -> list[str]:
        raise NotImplementedError

    def get_initial_concentration(self, dataset_model: DatasetModel) -> np.ndarray:
        raise NotImplementedError

    def get_k_matrix(self) -> KMatrix:
        raise NotImplementedError

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

        compartments = self.get_compartments(dataset_model)
        initial_concentration = self.get_initial_concentration(dataset_model)
        k_matrix = self.get_k_matrix()

        # the rates are the eigenvalues of the k matrix
        rates = k_matrix.rates(compartments, initial_concentration)

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
        matrix = matrix @ k_matrix.a_matrix(compartments, initial_concentration)

        # done
        return compartments, matrix

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        global_dimension = dataset_model.get_global_dimension()
        name = "images" if global_dimension == "pixel" else "spectra"
        decay_megacomplexes = [
            m for m in dataset_model.megacomplex if isinstance(m, DecayMegacomplexBase)
        ]

        species_dimension = "decay_species" if as_global else "species"
        if species_dimension not in dataset.coords:
            # We are the first Decay complex called and add SAD for all decay megacomplexes
            all_species = []
            for megacomplex in decay_megacomplexes:
                for species in megacomplex.get_compartments(dataset_model):
                    if species not in all_species:
                        all_species.append(species)
            retrieve_species_associated_data(
                dataset_model,
                dataset,
                all_species,
                species_dimension,
                global_dimension,
                name,
                is_full_model,
                as_global,
            )
        if isinstance(dataset_model.irf, IrfMultiGaussian) and "irf" not in dataset:
            retrieve_irf(dataset_model, dataset, global_dimension)

        if not is_full_model:
            multiple_complexes = len(decay_megacomplexes) > 1
            retrieve_decay_associated_data(
                self,
                dataset_model,
                dataset,
                global_dimension,
                name,
                multiple_complexes,
            )
