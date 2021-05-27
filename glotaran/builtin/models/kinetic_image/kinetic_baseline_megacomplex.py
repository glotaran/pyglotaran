"""This package contains the kinetic megacomplex item."""
from __future__ import annotations

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex
from glotaran.model import model_attribute


@model_attribute()
class KineticBaselineMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        model,
        dataset_descriptor: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        size = axis[model.model_dimension].size
        compartments = [f"{dataset_descriptor.label}_baseline"]
        matrix = np.ones((size, 1), dtype=np.float64)
        return (compartments, matrix)
