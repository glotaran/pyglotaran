from __future__ import annotations

from uuid import uuid4

from attr import ib
from attrs import Attribute
from attrs import define
from attrs import field
from attrs import make_class

from glotaran.model_new.dataset_group import DatasetGroupModel
from glotaran.model_new.dataset_model import DatasetModel
from glotaran.model_new.item import ModelItemTyped
from glotaran.model_new.item import infer_model_item_type_from_attribute
from glotaran.model_new.item import model_items
from glotaran.model_new.megacomplex import Megacomplex
from glotaran.model_new.weight import Weight

DEFAULT_DATASET_GROUP = "default"


def _load_item_from_dict(cls, value: any, extra: dict[str, any] = {}) -> any:
    if isinstance(value, dict):
        if issubclass(cls, ModelItemTyped):
            item_type = value["type"]
            cls = cls.get_item_type_class(item_type)
        value = cls(**(value | extra))
    return value


def _load_model_items_from_dict(cls, item_dict: dict[str, any]) -> dict[str, any]:
    return {
        label: _load_item_from_dict(cls, value, extra={"label": label})
        for label, value in item_dict.items()
    }


def _load_global_items_from_dict(cls, item_list: list[any]) -> list[any]:
    return [_load_item_from_dict(cls, value) for value in item_list]


def _add_default_dataset_group(
    dataset_groups: dict[str, DatasetGroupModel]
) -> dict[str, DatasetGroupModel]:
    dataset_groups = _load_model_items_from_dict(DatasetGroupModel, dataset_groups)
    if DEFAULT_DATASET_GROUP not in dataset_groups:
        dataset_groups[DEFAULT_DATASET_GROUP] = DatasetGroupModel()
    return dataset_groups


def _model_item_attribute(model_item_type: type):
    return ib(
        type=dict[str, model_item_type],
        factory=dict,
        converter=lambda value: _load_model_items_from_dict(model_item_type, value),
    )


def _infer_default_megacomplex() -> str:
    return next(Megacomplex.get_item_types())


@define(kw_only=True)
class Model:

    dataset_groups: dict[str, DatasetGroupModel] = field(
        factory=dict, converter=_add_default_dataset_group
    )

    dataset: dict[str, DatasetModel]

    megacomplex: dict[str, Megacomplex] = field(
        factory=dict,
        converter=lambda value: _load_model_items_from_dict(Megacomplex, value),
    )

    weights: list[Weight] = field(
        factory=list, converter=lambda value: _load_global_items_from_dict(Weight, value)
    )

    @classmethod
    def create_class(cls, items: dict[str, Attribute]) -> Model:
        cls_name = f"GlotaranModel_{str(uuid4()).replace('-','_')}"
        return make_class(cls_name, items, bases=(cls,))

    @classmethod
    def create_class_from_megacomplexes(cls, megacomplexes: list[Megacomplex]) -> Model:
        items: dict[str, Attribute] = {}
        dataset_types = set()
        for megacomplex in megacomplexes:
            if dataset_model_type := megacomplex.get_dataset_model_type():
                dataset_types |= {
                    dataset_model_type,
                }
            for model_item in model_items(megacomplex):
                model_item_type = infer_model_item_type_from_attribute(model_item)
                items[model_item.name] = _model_item_attribute(model_item_type)

        if len(dataset_types) == 0:
            dataset_types = (DatasetModel,)
        items["dataset"] = _model_item_attribute(
            make_class(
                f"GlotaranModel_{str(uuid4()).replace('-','_')}",
                [],
                bases=tuple(dataset_types),
                collect_by_mro=True,
            )
        )

        return cls.create_class(items)
