from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import create_model
from pydantic import validator
from pydantic.fields import FieldInfo

from glotaran.model.data_model import DataModel
from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import ItemIssue
from glotaran.model.errors import ModelItemIssue
from glotaran.model.errors import ParameterIssue
from glotaran.model.item import META_VALIDATOR
from glotaran.model.item import Item
from glotaran.model.item import LibraryItem
from glotaran.model.item import LibraryItemT
from glotaran.model.item import LibraryItemTyped
from glotaran.model.item import get_structure_and_type_from_field
from glotaran.model.item import iterate_library_item_fields
from glotaran.model.item import iterate_library_item_types
from glotaran.model.item import iterate_parameter_fields
from glotaran.model.megacomplex import Megacomplex
from glotaran.parameter import Parameters


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
    def create(cls, item_types: Sequence[type[LibraryItem]]) -> type[Library]:
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
    def create_for_megacomplexes(
        cls, megacomplexes: list[type[Megacomplex]] | None = None
    ) -> type[Library]:

        megacomplexes = megacomplexes or Megacomplex.__item_types__
        data_models = [
            m.data_model_type
            for m in filter(lambda m: m.data_model_type is not None, megacomplexes)
        ] + [DataModel]
        items = {
            library_item
            for item in chain(megacomplexes, data_models)
            for library_item in iterate_library_item_types(item)
        }
        return cls.create(items)

    @classmethod
    def from_dict(
        cls, obj: dict[str, Any], megacomplexes: list[type[Megacomplex]] | None = None
    ) -> Library:
        return cls.create_for_megacomplexes(megacomplexes).parse_obj(obj)

    def get_item(self, item_type: str | type[LibraryItem], label: str) -> LibraryItem:

        if not isinstance(item_type, str):
            item_type = item_type.get_library_name()

        if not hasattr(self, item_type):
            raise GlotaranModelError(f"Cannot get item of unknown type '{item_type}'.")
        item = getattr(self, item_type).get(label, None)
        if item is None:
            raise GlotaranModelError(
                f"Library contains no item of type '{item_type}' with label '{label}'."
            )
        return item

    def get_data_model_for_megacomplexes(self, megacomplex_labels: list[str]) -> type[DataModel]:
        data_model_cls_name = f"GlotaranDataModel_{str(uuid4()).replace('-','_')}"
        megacomplexes = {type(self.get_item(Megacomplex, label)) for label in megacomplex_labels}
        data_models = [
            m.data_model_type
            for m in filter(lambda m: m.data_model_type is not None, megacomplexes)
        ] + [DataModel]
        return create_model(data_model_cls_name, __base__=tuple(data_models))

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

    def validate_library_item_value(
        self,
        item_type: type[LibraryItem],
        value: Item,
        issues: list[ItemIssue],
        parameters: Parameters | None,
    ):
        try:
            issues += self.validate_item(self.get_item(item_type, value), parameters=parameters)
        except GlotaranModelError:
            issues.append(ModelItemIssue(item_type.get_library_name(), value))

    def validate_item(self, item: Item, parameters: Parameters | None = None) -> list[ItemIssue]:
        issues = []
        for field in iterate_library_item_fields(item):
            value = getattr(item, field.name)
            if value is None:
                continue
            if META_VALIDATOR in field.field_info.extra:
                issues += field.field_info.extra[META_VALIDATOR](value, item, self, parameters)
            structure, item_type = get_structure_and_type_from_field(field)
            if structure is None:
                self.validate_library_item_value(item_type, value, issues, parameters)
            else:
                values = value.values() if structure is dict else value
                for v in values:
                    self.validate_library_item_value(item_type, v, issues, parameters)

        if parameters is not None:
            for field in iterate_parameter_fields(item):
                value = getattr(item, field.name)
                if value is None:
                    continue
                structure, _ = get_structure_and_type_from_field(field)
                if structure is None:
                    if not parameters.has(value):
                        issues += [ParameterIssue(value)]
                else:
                    values = value.values() if structure is dict else value
                    issues += [ParameterIssue(v) for v in values if not parameters.has(v)]

        return issues
