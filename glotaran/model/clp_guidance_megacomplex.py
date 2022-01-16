from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex

if TYPE_CHECKING:
    from glotaran.model.dataset_model import DatasetModel


@megacomplex()
class ClpGuidanceMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):
        model_axis = dataset_model.get_model_axis()
        clp_label = [dataset_model.clp_guidance]
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
            dataset["estimated_clp_guidance"] = dataset.clp.sel(
                clp_label=dataset_model.clp_guidance
            )
