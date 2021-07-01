from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import xarray as xr
from typing_inspect import get_args
from typing_inspect import is_generic_type

from glotaran.model import DatasetDescriptor
from glotaran.model.item import model_item
from glotaran.model.item import model_item_typed

if TYPE_CHECKING:
    from typing import Any


def create_model_megacomplex_type(
    megacomplex_types: dict[str, Megacomplex], default_type: str = None
) -> type:
    @model_item_typed(types=megacomplex_types, default_type=default_type)
    class ModelMegacomplex:
        """This class holds all Megacomplex types defined by a model."""

    return ModelMegacomplex


def megacomplex(
    *,
    dimension: str,
    model_items: dict[str, dict[str, Any]] = None,
    properties: Any | dict[str, dict[str, Any]] = None,
    dataset_model_items: dict[str, dict[str, Any]] = None,
    dataset_properties: Any | dict[str, dict[str, Any]] = None,
):
    """The `@megacomplex` decorator is intended to be used on subclasses of
    :class:`glotaran.model.Megacomplex`. It registers the megacomplex model
    and makes it available in analysis models.
    """
    properties = properties if properties is not None else {}
    properties["dimension"] = {"type": str, "default": dimension}

    if model_items is None:
        model_items = {}
    else:
        model_items, properties = _add_model_items_to_properties(model_items, properties)

    dataset_properties = dataset_properties if dataset_properties is not None else {}
    if dataset_model_items is None:
        dataset_model_items = {}
    else:
        dataset_model_items, dataset_properties = _add_model_items_to_properties(
            dataset_model_items, dataset_properties
        )

    def decorator(cls):

        setattr(cls, "_glotaran_megacomplex_model_items", model_items)
        setattr(cls, "_glotaran_megacomplex_dataset_model_items", dataset_model_items)
        setattr(cls, "_glotaran_megacomplex_dataset_properties", dataset_properties)

        return model_item(properties=properties, has_type=True)(cls)

    return decorator


def _add_model_items_to_properties(model_items: dict, properties: dict) -> tuple[dict, dict]:
    for name, item in model_items.items():
        item_type = item["type"] if isinstance(item, dict) else item
        property_type = str

        if is_generic_type(item_type):
            if item_type._name == "List":
                property_type = List[str]
                item_type = get_args(item_type)[0]
            elif item_type._name == "Dict":
                property_type = Dict[str, str]
                item_type = get_args(item_type)[1]

        property_dict = item.copy() if isinstance(item, dict) else {}
        property_dict["type"] = property_type
        properties[name] = property_dict
        model_items[name] = item_type
    return model_items, properties


def _create_dataset_model_proper(dataset_model_items: dict) -> dict:
    return {
        name: {"type": item} if not isinstance(item, dict) else item
        for name, item in dataset_model_items()
    }


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

    @classmethod
    def glotaran_dataset_model_items(cls) -> str:
        return cls._glotaran_megacomplex_dataset_model_items

    @classmethod
    def glotaran_dataset_properties(cls) -> str:
        return cls._glotaran_megacomplex_dataset_properties
