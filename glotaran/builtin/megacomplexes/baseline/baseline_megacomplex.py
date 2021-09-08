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
        model_axis = dataset_model.get_model_axis()
        clp_label = [f"{dataset_model.label}_baseline"]
        matrix = np.ones((model_axis.size, 1), dtype=np.float64)
        return clp_label, matrix

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return False

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        if not is_full_model:
            dataset["baseline"] = dataset.clp.sel(clp_label=f"{dataset_model.label}_baseline")
