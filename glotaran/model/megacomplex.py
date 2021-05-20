from __future__ import annotations

import numpy as np

from glotaran.model.dataset_descriptor import DatasetDescriptor


class Megacomplex:
    def calculate_matrix(
        self,
        dataset_descriptor: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        raise NotImplementedError
