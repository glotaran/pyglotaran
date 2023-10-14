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
    register_as = "clp-guide"  # type:ignore[pydantic-field]
    exclusive = True  # type:ignore[pydantic-field]
    target: str

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        return [self.target], np.ones((1, 1), dtype=np.float64)
