from __future__ import annotations

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute


@model_attribute(has_type=True)
class Megacomplex:
    def calculate_matrix(
        self,
        model,
        dataset_descriptor: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        raise NotImplementedError
