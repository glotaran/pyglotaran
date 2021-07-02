from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex


#  @megacomplex("time")
class KineticBaselineMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetDescriptor,
        indices: dict[str, int],
        **kwargs,
    ):
        model_dimension = dataset_model.get_model_dimension()
        model_axis = dataset_model.get_coords()[model_dimension]
        compartments = [f"{dataset_model.label}_baseline"]
        matrix = np.ones((model_axis.size, 1), dtype=np.float64)
        return xr.DataArray(
            matrix, coords=((model_dimension, model_axis.data), ("clp_label", compartments))
        )

    def index_dependent(self, dataset: DatasetDescriptor) -> bool:
        return False
