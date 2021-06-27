from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr

from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex


@megacomplex(
    "spectral",
    properties={
        "shape": Dict[str, str],
    },
)
class SpectralMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetDescriptor,
        indices: dict[str, int],
        **kwargs,
    ):

        compartments = []
        for compartment in self.shape:
            if compartment in compartments:
                raise ModelError(f"More then one shape defined for compartment '{compartment}'")
            compartments.append(compartment)

        model_dimension = dataset_model.get_model_dimension()
        model_axis = dataset_model.get_coords()[model_dimension]

        dim1 = model_axis.size
        dim2 = len(self.shape)
        matrix = np.zeros((dim1, dim2))

        for i, shape in enumerate(self.shape.values()):
            matrix[:, i] += shape.calculate(model_axis.values)
        return xr.DataArray(
            matrix, coords=((model_dimension, model_axis.data), ("clp_label", compartments))
        )

    def index_dependent(self, dataset: DatasetDescriptor) -> bool:
        return False
