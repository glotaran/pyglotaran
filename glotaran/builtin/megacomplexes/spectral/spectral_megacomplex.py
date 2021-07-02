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
    model_items={
        "shape": Dict[str, SpectralShape],
    },
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

        model_dimension = dataset_model.get_model_dimension()
        model_axis = dataset_model.get_coordinates()[model_dimension]

        dim1 = model_axis.size
        dim2 = len(self.shape)
        matrix = np.zeros((dim1, dim2))

        for i, shape in enumerate(self.shape.values()):
            matrix[:, i] += shape.calculate(model_axis.values)
        return xr.DataArray(
            matrix, coords=((model_dimension, model_axis.data), ("clp_label", compartments))
        )

    def index_dependent(self, dataset: DatasetModel) -> bool:
        return False

    def finalize_data(self, dataset_model: DatasetModel, data: xr.Dataset):
        if "species" in data.coords:
            return

        species = []
        for megacomplex in dataset_model.megacomplex:  # noqa F402
            if isinstance(megacomplex, SpectralMegacomplex):
                species += [
                    compartment for compartment in megacomplex.shape if compartment not in species
                ]

        data.coords["species"] = species
        data["species_spectra"] = (
            (
                dataset_model.get_model_dimension(),
                "species",
            ),
            data.matrix.sel(clp_label=species).values,
        )
        data["species_associated_concentrations"] = (
            (
                dataset_model.get_global_dimension(),
                "species",
            ),
            data.clp.sel(clp_label=species).data,
        )
