from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import megacomplex


@megacomplex(unique=True, register_as="baseline")
class BaselineMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):
        model_dimension = dataset_model.get_model_dimension()
        model_axis = dataset_model.get_coordinates()[model_dimension]
        clp_label = [f"{dataset_model.label}_baseline"]
        matrix = np.ones((model_axis.size, 1), dtype=np.float64)
        return xr.DataArray(
            matrix, coords=((model_dimension, model_axis.data), ("clp_label", clp_label))
        )

    def index_dependent(self, dataset: DatasetModel) -> bool:
        return False

    def finalize_data(self, dataset_model: DatasetModel, data: xr.Dataset):
        data[f"{dataset_model.label}_baseline"] = data.clp.sel(clp_label="baseline")
