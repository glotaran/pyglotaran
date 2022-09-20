from inspect import getmro
from types import NoneType
from types import UnionType
from typing import ClassVar
from typing import Generator
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

from attrs import NOTHING
from attrs import Attribute
from attrs import define
from attrs import fields
from attrs import resolve_types

from glotaran.parameter import Parameter

T = TypeVar("T")

ParameterType: TypeAlias = Parameter | str
ModelItemType: TypeAlias = T | str


class Item:
    pass


def item(cls):
    parent = getmro(cls)[1]
    cls = define(kw_only=True, slots=False)(cls)
    resolve_types(cls)
    if parent is ModelItemTyped:
        cls.__model_item_types__ = {}
    elif issubclass(cls, ModelItemTyped):
        cls._register_item_class()
    return cls


@define(kw_only=True)
class ModelItem(Item):
    label: str


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


def model_items(item: Item) -> Generator[Attribute, None, None]:
    for attr in fields(item):
        _, item_type = strip_type_and_structure_from_attribute(attr)
        if issubclass(item_type, ModelItem):
            yield attr


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
