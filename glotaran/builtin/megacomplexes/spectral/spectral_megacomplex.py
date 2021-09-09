from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.spectral.shape import SpectralShape
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex


@megacomplex(
    dimension="spectral",
    dataset_properties={
        "spectral_axis_inverted": {"type": bool, "default": False},
        "spectral_axis_scale": {"type": float, "default": 1},
    },
    model_items={
        "shape": Dict[str, SpectralShape],
    },
    register_as="spectral",
)
class SpectralMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):

        compartments = []
        for compartment in self.shape:
            if compartment in compartments:
                raise ModelError(f"More then one shape defined for compartment '{compartment}'")
            compartments.append(compartment)

        model_axis = dataset_model.get_model_axis()
        if dataset_model.spectral_axis_inverted:
            model_axis = dataset_model.spectral_axis_scale / model_axis
        elif dataset_model.spectral_axis_scale != 1:
            model_axis = model_axis * dataset_model.spectral_axis_scale

        dim1 = model_axis.size
        dim2 = len(self.shape)
        matrix = np.zeros((dim1, dim2))

        for i, shape in enumerate(self.shape.values()):
            matrix[:, i] += shape.calculate(model_axis)

        return compartments, matrix

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return False

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        species_dimension = "spectral_species" if as_global else "species"
        if species_dimension in dataset.coords:
            return

        species = []
        megacomplexes = (
            dataset_model.global_megacomplex if as_global else dataset_model.megacomplex
        )
        for m in megacomplexes:
            if isinstance(m, SpectralMegacomplex):
                species += [compartment for compartment in m.shape if compartment not in species]

        dataset.coords[species_dimension] = species
        matrix = dataset.global_matrix if as_global else dataset.matrix
        clp_dim = "global_clp_label" if as_global else "clp_label"
        dataset["species_spectra"] = (
            (
                dataset_model.get_model_dimension()
                if not as_global
                else dataset_model.get_global_dimension(),
                species_dimension,
            ),
            matrix.sel({clp_dim: species}).values,
        )
        if not is_full_model:
            dataset["species_associated_concentrations"] = (
                (
                    dataset_model.get_global_dimension(),
                    species_dimension,
                ),
                dataset.clp.sel(clp_label=species).data,
            )
