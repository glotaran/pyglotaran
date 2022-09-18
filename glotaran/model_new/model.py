from __future__ import annotations

from uuid import uuid4

from attrs import define
from attrs import field
from attrs import make_class

from glotaran.model_new.dataset_group import DatasetGroupModel
from glotaran.model_new.weight import Weight

DEFAULT_DATASET_GROUP = "default"


def _load_item_from_dict(cls, value: any) -> any:
    return cls(**value) if isinstance(value, dict) else value


def _load_model_items_from_dict(cls, item_dict: dict[str, any]) -> dict[str, any]:
    return {label: _load_item_from_dict(cls, value) for label, value in item_dict.items()}


def _load_global_items_from_dict(cls, item_list: list[any]) -> list[any]:
    return [_load_item_from_dict(cls, value) for value in item_list]


def _add_default_dataset_group(
    dataset_groups: dict[str, DatasetGroupModel]
) -> dict[str, DatasetGroupModel]:
    dataset_groups = _load_model_items_from_dict(DatasetGroupModel, dataset_groups)
    if DEFAULT_DATASET_GROUP not in dataset_groups:
        dataset_groups[DEFAULT_DATASET_GROUP] = DatasetGroupModel()
    return dataset_groups


@define
class Model:

    dataset_groups: dict[str, DatasetGroupModel] = field(
        factory=dict, converter=_add_default_dataset_group
    )

    weights: list[Weight] = field(
        factory=list, converter=lambda value: _load_global_items_from_dict(Weight, value)
    )

    @classmethod
    def create(cls, items: list) -> Model:
        cls_name = f"GlotaranModel_{str(uuid4()).replace('-','_')}"
        return make_class(cls_name, {}, bases=(cls,))
