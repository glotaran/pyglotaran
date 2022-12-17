from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import create_model
from pydantic import validator
from pydantic.fields import FieldInfo

from glotaran.model.item_new import Item
from glotaran.model.item_new import LibraryItem
from glotaran.model.item_new import LibraryItemT
from glotaran.model.item_new import LibraryItemTyped
from glotaran.model.item_new import get_structure_and_type_from_field
from glotaran.model.item_new import iterate_library_item_fields


def add_label_to_items(items: dict[str, Any]) -> dict[str, Any]:
    for label, item in items.items():
        if isinstance(item, dict):
            item["label"] = label
    return items


def create_field_type_and_info_for_item_type(
    item_type: type[LibraryItem],
) -> tuple[type, FieldInfo]:
    if issubclass(item_type, LibraryItemTyped):
        item_type = item_type.get_annotated_type()
    return (dict[str, item_type], FieldInfo(default_factory=dict))


class Library(BaseModel):
    @classmethod
    def create(cls, item_types: list[type[LibraryItem]]) -> Library:
        library_cls_name = f"GlotaranLibrary_{str(uuid4()).replace('-','_')}"
        library_fields = {
            it.get_library_name(): create_field_type_and_info_for_item_type(it)
            for it in item_types
        }
        library_validators = {
            "__add_label_to_items": validator(
                *list(library_fields.keys()), allow_reuse=True, pre=True
            )(add_label_to_items)
        }
        cls = create_model(
            library_cls_name, **library_fields, __validators__=library_validators, __base__=cls
        )
        return cls

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Library:
        return cls.parse_obj(obj)

    def resolve_item_by_type_and_value(
        self, item_type: LibraryItemT, value: str | LibraryItem
    ) -> LibraryItem:
        if isinstance(value, str):
            value = getattr(self, item_type.get_library_name())[value]
            value = self.resolve_item(value)
        return value

    def resolve_item(self, item: Item) -> Item:
        resolved = {}
        for field in iterate_library_item_fields(item):
            value = getattr(item, field.name)
            if value is None:
                continue
            structure, item_type = get_structure_and_type_from_field(field)
            if structure is None:
                resolved[field.name] = self.resolve_item_by_type_and_value(item_type, value)
            elif structure is list:
                resolved[field.name] = [
                    self.resolve_item_by_type_and_value(item_type, v) for v in value
                ]
            elif structure is dict:
                resolved[field.name] = {
                    k: self.resolve_item_by_type_and_value(item_type, v) for k, v in value.items()
                }
        return item.copy(update=resolved)
