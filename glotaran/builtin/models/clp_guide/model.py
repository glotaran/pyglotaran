from __future__ import annotations

from typing import Literal

import numpy as np

from glotaran.model import DataModel
from glotaran.model import Model


class ClpGuideModel(Model):
    type: Literal["clp-guide"]
    register_as = "clp-guide"
    target: str
    exclusive = True

    def calculate_matrix(
        self,
        dataset_model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):
        return [self.target], np.ones((1, 1), dtype=np.float64)
