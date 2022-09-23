from __future__ import annotations

from inspect import getmro
from textwrap import indent
from types import NoneType
from types import UnionType
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Generator
from typing import Iterator
from typing import Type
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

from attrs import NOTHING
from attrs import Attribute
from attrs import define
from attrs import field
from attrs import fields
from attrs import resolve_types

from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from glotaran.model_new.model import Model


META_ALIAS = "__glotaran_alias__"
META_VALIDATOR = "__glotaran_validator__"


class ItemIssue:
    def to_string(self) -> str:
        raise NotImplementedError

    def __rep__(self) -> str:
        return self.to_string()


class ModelItemIssue(ItemIssue):
    def __init__(self, item_name: str, label: str):
        self._item_name = item_name
        self._label = label

    def to_string(self) -> str:
        return f"Missing model item '{self.item_name}' with label '{self.label}'."


class ParameterIssue(ItemIssue):
    def __init__(self, label: str):
        self._label = label

    def to_string(self) -> str:
        return f"Missing parameter with label '{self.label}'."


class Item:
    pass


def item_to_markdown(
    item: Item, parameters: ParameterGroup = None, initial_parameters: ParameterGroup = None
) -> MarkdownStr:
    md = "\n"
    for attr in fields(item.__class__):
        name = attr.name
        value = getattr(item, name)
        if value is None:
            continue

        structure, item_type = strip_type_and_structure_from_attribute(attr)
        if item_type is Parameter and parameters is not None:
            if structure is dict:
                value = {
                    k: parameters.get(v).markdown(parameters, initial_parameters)
                    for k, v in value.items()
                }
            elif structure is list:
                value = [parameters.get(v).markdown(parameters, initial_parameters) for v in value]
            else:
                value = parameters.get(value).markdown(parameters, initial_parameters)

        property_md = indent(f"* *{name.replace('_', ' ').title()}*: {value}\n", "  ")

        md += property_md

    return MarkdownStr(md)


def iterate_attributes_of_type(item: Item, attr_type: type) -> Generator[Attribute, None, None]:
    for attr in fields(item):
        _, item_type = strip_type_and_structure_from_attribute(attr)
        if isinstance(item_type, Type) and issubclass(item_type, attr_type):
            yield attr


def model_attributes(item: Item, with_alias: bool = True) -> Generator[Attribute, None, None]:
    for attr in iterate_attributes_of_type(item, ModelItem):
        if with_alias or META_ALIAS not in attr.metadata:
            yield attr


def parameter_attributes(item: Item) -> Generator[Attribute, None, None]:
    return iterate_attributes_of_type(item, Parameter)


def iterate_names_and_labels(
    item: Item, attributes: Generator[Attribute, None, None]
) -> Generator[tuple(str, str), None, None]:
    for attr in attributes:
        structure, _ = strip_type_and_structure_from_attribute(attr)
        value = getattr(item, attr.name)
        name = attr.metadata.get(META_ALIAS, attr.name)

        if not value:
            continue

        if structure is dict:
            for v in value.values():
                yield name, v

        elif structure is list:
            for v in value:
                yield name, v

        else:
            yield name, value


def iterate_model_item_names_and_labels(item: Item) -> Generator[tuple(str, str), None, None]:
    return iterate_names_and_labels(item, model_attributes(item.__class__))


def iterate_parameter_names_and_labels(item: Item) -> Generator[tuple(str, str), None, None]:
    return iterate_names_and_labels(item, parameter_attributes(item.__class__))


def fill_item_attributes(
    item: item, iterator: Iterator[Attribute], fill_function: Callable
) -> list[ItemIssue]:
    for attr in iterator:
        value = getattr(item, attr.name)

        structure, _ = strip_type_and_structure_from_attribute(attr)
        name = attr.metadata.get(META_ALIAS, attr.name)
        if structure is dict:
            value = {
                k: fill_function(name, v) if isinstance(v, str) else v for k, v in value.items()
            }
        elif structure is list:
            value = [fill_function(name, v) if isinstance(v, str) else v for v in value]
        else:
            value = fill_function(name, value) if isinstance(value, str) else value

        setattr(item, attr.name, value)


def fill_item(item: item, model: Model, parameters: ParameterGroup) -> Item:
    fill_item_model_attributes(item, model, parameters)
    fill_item_parameter_attributes(item, parameters)
    return item


def fill_item_model_attributes(item: item, model: Model, parameters: ParameterGroup):
    fill_item_attributes(
        item,
        model_attributes(item.__class__),
        lambda name, label: fill_item(getattr(model, name)[label], model, parameters),
    )


def fill_item_parameter_attributes(item: item, parameters: ParameterGroup) -> list[ItemIssue]:
    fill_item_attributes(
        item, parameter_attributes(item.__class__), lambda name, label: parameters.get(label)
    )


def get_item_model_issues(item: item, model: Model) -> list[ItemIssue]:
    return [
        ModelItemIssue(name, label)
        for name, label in iterate_model_item_names_and_labels(item)
        if label not in getattr(model, name)
    ]


def get_item_parameter_issues(item: item, parameters: ParameterGroup) -> list[ItemIssue]:
    return [
        ParameterIssue(label)
        for name, label in iterate_parameter_names_and_labels(item)
        if not parameters.has(label)
    ]


def get_item_validator_issues(
    item: item, model: Model, parameters: ParameterGroup | None = None
) -> list[ItemIssue]:
    issues = []
    for name, validator in [
        (attr.name, attr.metadata[META_VALIDATOR])
        for attr in fields(item.__class__)
        if META_VALIDATOR in attr.metadata
    ]:
        issues += validator(getattr(item, name), model, parameters)

    return issues


def get_item_issues(
    *, item: Item, model: Model, parameters: ParameterGroup | None = None
) -> list[ItemIssue]:
    issues = get_item_model_issues(item, model)
    issues += get_item_validator_issues(item, model, parameters)
    if parameters is not None:
        issues += get_item_parameter_issues(item, parameters)
    return issues


@define(kw_only=True)
class ModelItem(Item):
    label: str


T = TypeVar("ModelItemT", bound="ModelItem")

ParameterType: TypeAlias = Parameter | str
ModelItemType: TypeAlias = T | str


@define(kw_only=True)
class ModelItemTyped(ModelItem):
    type: str
    __model_item_types__: ClassVar[dict[str, type]]

    @classmethod
    def _register_item_class(cls):
        item_type = cls.get_item_type()
        if item_type is not NOTHING:
            cls.__model_item_types__[item_type] = cls

    @classmethod
    def get_item_type(cls) -> str:
        return fields(cls).type.default

    @classmethod
    def get_item_types(cls) -> str:
        return cls.__model_item_types__.keys()

    @classmethod
    def get_item_type_class(cls, item_type: str) -> type:
        return cls.__model_item_types__[item_type]


def strip_type_and_structure_from_attribute(attr: Attribute) -> tuple[None | list | dict, type]:
    definition = attr.type
    definition = strip_option_type(definition)
    structure, definition = strip_struture_type(definition)
    definition = strip_option_type(definition, strip_type=str)
    return structure, definition


def strip_option_type(definition: type, strip_type: type = NoneType) -> type:
    if get_origin(definition) in [Union, UnionType] and strip_type in get_args(definition):
        definition = get_args(definition)[0]
    return definition


def strip_struture_type(definition: type) -> tuple[None | list | dict, type]:
    structure = get_origin(definition)
    if structure is list:
        definition = get_args(definition)[0]
    elif structure is dict:
        definition = get_args(definition)[1]
    else:
        structure = None

    return structure, definition


def item(cls):
    parent = getmro(cls)[1]
    cls = define(kw_only=True, slots=False)(cls)
    if parent is ModelItemTyped:
        cls.__model_item_types__ = {}
    elif issubclass(cls, ModelItemTyped):
        cls._register_item_class()
    resolve_types(cls)
    return cls


def attribute(
    *,
    alias: str | None = None,
    default: any = NOTHING,
    validator: Callable[[ModelItem, Model, ParameterGroup | None], list[ItemIssue]] | None = None,
) -> Attribute:
    metadata = {}
    if alias is not None:
        metadata[META_ALIAS] = alias
    if validator is not None:
        metadata[META_VALIDATOR] = validator
    return field(default=default, metadata=metadata)
