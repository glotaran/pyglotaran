from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import xarray as xr

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute

if TYPE_CHECKING:
    from typing import Any


def megacomplex(
    dimension: str,
    properties: Any | dict[str, dict[str, Any]] = None,
    items: dict[str, dict[str, Any]] = None,
    dataset_attributes: dict[str, dict[str, Any]] = None,
):
    """The `@megacomplex` decorator is intended to be used on subclasses of
    :class:`glotaran.model.Megacomplex`. It registers the megacomplex model
    and makes it available in analysis models.
    """
    # TODO: this is temporary and will change in follow up PR
    properties = properties if properties is not None else {}
    properties["dimension"] = {"type": str, "default": dimension}

    items = items if items is not None else {}
    for name, item in items.items():
        property_type = str

        if hasattr(item, "_name") and item._name == "List":
            property_type = List[str]

        elif hasattr(item, "_name") and item._name == "Dict":
            property_type = Dict[str, str]

        property_dict = item.copy() if isinstance(item, dict) else {}
        property_dict["type"] = property_type
        properties[name] = property_dict

    def decorator(cls):

        setattr(cls, "_glotaran_megacomplex_model_items", items)

        return model_attribute(properties=properties, has_type=True)(cls)

    return decorator


class Megacomplex:
    """A base class for megacomplex models.

    Subclasses must overwrite :method:`glotaran.model.Megacomplex.calculate_matrix`
    and :method:`glotaran.model.Megacomplex.index_dependent`.
    """

    def calculate_matrix(
        self,
        dataset_model: DatasetDescriptor,
        indices: dict[str, int],
        **kwargs,
    ) -> xr.DataArray:
        raise NotImplementedError

    def index_dependent(self, dataset: DatasetDescriptor) -> bool:
        raise NotImplementedError

    @classmethod
    def glotaran_model_items(cls) -> str:
        return cls._glotaran_megacomplex_model_items
