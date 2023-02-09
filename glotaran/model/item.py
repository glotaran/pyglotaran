"""This module contains the item classes and helper functions."""
import contextlib
import typing
from functools import cache
from inspect import getmro
from inspect import isclass
from types import NoneType
from types import UnionType
from typing import Annotated
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Generator
from typing import Type
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic.fields import ModelField
from pydantic.fields import Undefined

from glotaran.model.errors import ItemIssue
from glotaran.model.errors import ParameterIssue
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters

ItemT = TypeVar("ItemT", bound="Item")

ParameterType: TypeAlias = Parameter | float | str

META_VALIDATOR = "__glotaran_validator__"


class ItemAttribute(FieldInfo):
    """An attribute for items.

    A thin wrapper around pydantic.fields.FieldInfo.
    """

    def __init__(
        self,
        *,
        description: str,
        default: Any = Undefined,
        factory: Callable[[], Any] | None = None,
        validator: Callable | None = None,
    ):
        """Create an attribute for an item.

        Parameters
        ----------
        description: str
            The description of the attribute as shown in the help.
        default: Any
            The default value of the attribute.
        factory: Callable[[], Any] | None
            A factory function for the attribute.
        validator: Callable[[Any, Item, Model, Parameters | None], list[ItemIssue]] | None
            A validator function for the attribute.
        """
        metadata: dict[str, Any] = {}
        if validator is not None:
            metadata[META_VALIDATOR] = validator
        super().__init__(
            default=default, default_factory=factory, description=description, **metadata
        )


def Attribute(
    *,
    description: str,
    default: Any = Undefined,
    factory: Callable[[], Any] | None = None,
    validator: Callable | None = None,
) -> Any:
    """Create an attribute for an item.

    Parameters
    ----------
    description: str
        The description of the attribute as shown in the help.
    default: Any
        The default value of the attribute.
    factory: Callable[[], Any] | None
        A factory function for the attribute.
    validator: Callable[[Any, Item, Model, Parameters | None], list[ItemIssue]] | None
        A validator function for the attribute.

    Returns
    -------
    Any
    """
    return ItemAttribute(
        description=description, default=default, factory=factory, validator=validator
    )


class Item(BaseModel):
    """A baseclass for items."""

    class Config:
        """Config for pydantic.BaseModel."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class TypedItem(Item):
    """An item with a type."""

    type: str
    __item_types__: ClassVar[list[Type[Item]]]

    def __init_subclass__(cls):
        """Create an item from a class."""
        if cls.__qualname__ == "LibraryItemTyped":
            return
        parent = getmro(cls)[1]
        if parent is TypedItem:
            assert issubclass(cls, TypedItem)
            cls.__item_types__ = []
        elif issubclass(cls, TypedItem):
            cls.__item_types__.append(cls)

    @classmethod
    def get_annotated_type(cls) -> object:
        """Get the annotated type for discrimination.

        Returns
        -------
        object
        """
        return Annotated[Union[tuple(cls.__item_types__)], Field(discriminator="type")]


@cache
def get_structure_and_type_from_field(field: ModelField) -> tuple[None | list | dict, type]:
    """Get the structure and type from a field.

    Parameters
    ----------
    field: ModelField
        The field.

    Returns
    -------
    tuple[None | list | dict, type]:
        The structure and type as atuple.
    """
    definition = strip_option_type_from_definition(field.annotation)
    structure, definition = strip_structure_type_from_definition(definition)
    definition = strip_option_type_from_definition(definition, strip_type=str)
    return structure, definition


def strip_option_type_from_definition(definition: type, strip_type: type = NoneType) -> type:
    """Strip the type if the definition is an option.

    Parameters
    ----------
    definition: type
        The definition.
    strip_type: type
        The type which should be removed from the option.

    Returns
    -------
    type
    """
    args = list(get_args(definition))
    if get_origin(definition) in [Union, UnionType] and strip_type in args:
        args.remove(strip_type)
        definition = args[0]
    return definition


def strip_structure_type_from_definition(definition: type) -> tuple[None | list | dict, type]:
    """Strip the structure from a definition.

    Parameters
    ----------
    definition: type
        The definition.

    Returns
    -------
    tuple[None | list | dict, type]:
        The structure and the type.
    """
    structure = get_origin(definition)
    if structure is list:
        definition = get_args(definition)[0]
    elif structure is dict:
        definition = get_args(definition)[1]
    else:
        structure = None

    return structure, definition


def iterate_fields_of_type(
    item: type[Item], field_type: type
) -> Generator[ModelField, None, None]:
    """Iterate over all fields of the given types.

    Parameters
    ----------
    item: type[Item]
        The item type.
    field_type: type
        The field type.

    Yields
    ------
    ModelField
        The matching attributes.
    """
    for field in item.__fields__.values():
        _, item_type = get_structure_and_type_from_field(field)
        with contextlib.suppress(TypeError):
            # issubclass does for some reason not work with e.g. tuple as item_type
            # and Parameter as attr_type
            if (
                hasattr(item_type, "__origin__")
                and issubclass(typing.get_origin(item_type), typing.Annotated)
                and typing.get_origin(typing.get_args(item_type)[0]) is typing.Union
            ):
                item_type = typing.get_args(typing.get_args(item_type)[0])[0]
            if isclass(item_type) and issubclass(item_type, field_type):
                yield field


def iterate_item_fields(item: type[Item]) -> Generator[ModelField, None, None]:
    """Iterate over all item fields.

    Parameters
    ----------
    item: type[Item]
        The item type.

    Yields
    ------
    ModelField
        The item fields.
    """
    yield from iterate_fields_of_type(item, Item)


def iterate_parameter_fields(item: type[Item]) -> Generator[ModelField, None, None]:
    """Iterate over all parameter fields.

    Parameters
    ----------
    item: type[Item]
        The item type.

    Yields
    ------
    ModelField
        The parameter fields.
    """
    yield from iterate_fields_of_type(item, Parameter)


def resolve_item_parameters(
    item: ItemT, parameters: Parameters, initial: Parameters | None = None
) -> ItemT:
    resolved = {}
    initial = initial or parameters

    def resolve_parameter(parameter: Parameter | float | str) -> Parameter:
        if isinstance(parameter, str):
            if not parameters.has(parameter):
                parameters.add(initial.get(parameter).copy())
            parameter = parameters.get(parameter)
        return parameter

    for field in iterate_parameter_fields(item):
        value = getattr(item, field.name)
        if value is None:
            continue
        structure, _ = get_structure_and_type_from_field(field)
        if structure is None:
            resolved[field.name] = resolve_parameter(value)
        elif structure is list:
            resolved[field.name] = [resolve_parameter(v) for v in value]
        elif structure is dict:
            resolved[field.name] = {k: resolve_parameter(v) for k, v in value.items()}

    for field in iterate_item_fields(item):
        value = getattr(item, field.name)
        if value is None:
            continue
        structure, item_type = get_structure_and_type_from_field(field)
        if structure is None:
            resolved[field.name] = resolve_item_parameters(value, parameters, initial)
        elif structure is list:
            resolved[field.name] = [resolve_item_parameters(v, parameters, initial) for v in value]
        elif structure is dict:
            resolved[field.name] = {
                k: resolve_item_parameters(v, parameters, initial) for k, v in value.items()
            }
    return item.copy(update=resolved)


def get_item_issues(item: Item, parameters: Parameters) -> list[ItemIssue]:
    issues = []
    for field in iterate_item_fields(item):
        value = getattr(item, field.name)
        if value is None:
            continue
        if META_VALIDATOR in field.field_info.extra:
            issues += field.field_info.extra[META_VALIDATOR](value, item, parameters)
        structure, item_type = get_structure_and_type_from_field(field)
        if structure is None:
            issues += get_item_issues(value, parameters)
        else:
            values = value.values() if structure is dict else value
            for v in values:
                issues += get_item_issues(v, parameters)

    for field in iterate_parameter_fields(item):
        value = getattr(item, field.name)
        if value is None:
            continue
        structure, _ = get_structure_and_type_from_field(field)
        if structure is None:
            if isinstance(value, str) and not parameters.has(value):
                issues += [ParameterIssue(value)]
        else:
            values = value.values() if structure is dict else value
            issues += [
                ParameterIssue(v) for v in values if isinstance(v, str) and not parameters.has(v)
            ]

    return issues
