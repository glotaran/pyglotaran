from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import megacomplex


@megacomplex(exclusive=True, register_as="clp-guide", properties={"target": str})
class ClpGuideMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):
        clp_label = [self.target]
        matrix = np.ones((1, 1), dtype=np.float64)
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
        pass
