from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute

if TYPE_CHECKING:
    from typing import Any


def megacomplex(
    dimension: str,
    properties: Any | dict[str, dict[str, Any]] = {},
    attributes: dict[str, dict[str, Any]] = {},
    dataset_attributes: dict[str, dict[str, Any]] = {},
):
    properties["dimension"] = {"type": str, "default": dimension}
    return model_attribute(properties=properties, has_type=True)


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
