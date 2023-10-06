from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from glotaran.model import DataModel
from glotaran.model import Element

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class ClpGuideElement(Element):
    type: Literal["clp-guide"]
    register_as = "clp-guide"
    target: str
    exclusive = True

    def calculate_matrix(
        self,
        dataset_model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        return [self.target], np.ones((1, 1), dtype=np.float64)
