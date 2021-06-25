"""This package contains the kinetic megacomplex item."""
from __future__ import annotations

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex
from glotaran.model import megacomplex


@megacomplex("time")
class KineticBaselineMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        model,
        dataset_model: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        size = axis[dataset_model.get_model_dimension()].size
        compartments = [f"{dataset_model.label}_baseline"]
        matrix = np.ones((size, 1), dtype=np.float64)
        return (compartments, matrix)

    def index_dependent(self, dataset: DatasetDescriptor) -> bool:
        return False
