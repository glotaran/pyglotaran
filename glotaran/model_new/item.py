from __future__ import annotations

from inspect import getmro
from types import NoneType
from types import UnionType
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Generator
from typing import Iterator
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

try:
    from typing import Self
except ImportError:
    Self = TypeVar("GlotaranItemT", bound="Item")

if TYPE_CHECKING:
    from glotaran.model_new.model import Model

from attrs import NOTHING
from attrs import Attribute
from attrs import define
from attrs import fields
from attrs import resolve_types

from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup


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


def iterate_attributes_of_type(item: Item, attr_type: type) -> Generator[Attribute, None, None]:
    for attr in fields(item):
        _, item_type = strip_type_and_structure_from_attribute(attr)
        if issubclass(item_type, attr_type):
            yield attr


def model_attributes(item: Item) -> Generator[Attribute, None, None]:
    return iterate_attributes_of_type(item, ModelItem)


def parameter_attributes(item: Item) -> Generator[Attribute, None, None]:
    return iterate_attributes_of_type(item, Parameter)


def iterate_names_and_labels(
    item: Item, attributes: Generator[Attribute, None, None]
) -> Generator[tuple(str, str), None, None]:
    for attr in attributes:
        structure, _ = strip_type_and_structure_from_attribute(attr)
        value = getattr(item, attr.name)

        if structure is dict:
            for v in value.values():
                yield attr.name, v

        elif structure is list:
            for v in value:
                yield attr.name, v

        else:
            yield attr.name, value


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
        if structure is dict:
            value = {
                k: fill_function(attr.name, v) if isinstance(v, str) else v
                for k, v in value.items()
            }
        elif structure is list:
            value = [fill_function(attr.name, v) if isinstance(v, str) else v for v in value]
        else:
            value = fill_function(attr.name, value) if isinstance(value, str) else value

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


def get_item_issues(
    *, item: Item, model: Model, parameters: ParameterGroup | None = None
) -> list[ItemIssue]:
    issues = get_item_model_issues(item, model)
    if parameters is not None:
        issues += get_item_parameter_issues(item, parameters)
    return issues


def item(cls):
    parent = getmro(cls)[1]
    cls = define(kw_only=True, slots=False)(cls)
    if parent is ModelItemTyped:
        cls.__model_item_types__ = {}
    elif issubclass(cls, ModelItemTyped):
        cls._register_item_class()
    resolve_types(cls)
    return cls


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
