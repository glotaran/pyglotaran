from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute

if TYPE_CHECKING:
    from typing import Any


def megacomplex(
    dimension: str,
    properties: Any | dict[str, dict[str, Any]] = None,
    attributes: dict[str, dict[str, Any]] = None,
    dataset_attributes: dict[str, dict[str, Any]] = None,
):
    """The `@megacomplex` decorator is intended to be used on subclasses of
    :class:`glotaran.model.Megacomplex`. It registers the megacomplex model
    and makes it available in analysis models.
    """

    # TODO: this is temporary and will change in follow up PR
    properties = properties if properties is not None else {}
    properties["dimension"] = {"type": str, "default": dimension}
    return model_attribute(properties=properties, has_type=True)


class Megacomplex:
    """A base class for megacomplex models.

    Subclasses must overwrite :method:`glotaran.model.Megacomplex.calculate_matrix`
    and :method:`glotaran.model.Megacomplex.index_dependent`.
    """

    def calculate_matrix(
        self,
        model,
        dataset_descriptor: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        raise NotImplementedError

    def index_dependent(self, dataset: DatasetDescriptor) -> bool:
        raise NotImplementedError
