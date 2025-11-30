"""This module contains the item classes and helper functions."""

from __future__ import annotations

import contextlib
import typing
from collections import UserDict
from functools import cache
from inspect import getmro
from inspect import isclass
from types import NoneType
from types import UnionType
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from glotaran.model.errors import ItemIssue
from glotaran.model.errors import ParameterIssue
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

ItemT = TypeVar("ItemT", bound="Item")

ParameterType: TypeAlias = Parameter | float | str

META_VALIDATOR = "__glotaran_validator__"


class GlotaranFieldMetadata(UserDict):
    """Container to hold glotaran field meta data."""

    @property
    def validator(self) -> Callable | None:
        """Glotaran validator function if defined, else None."""
        return self.get(META_VALIDATOR, None)


def extract_glotaran_field_metadata(info: FieldInfo) -> GlotaranFieldMetadata:
    """Extract glotaran metadata from field info metadata list.

    Parameters
    ----------
    info : FieldInfo
        Field info to for glotaran metadata in.

    Returns
    -------
    GlotaranFieldMetadata
        Glotaran meta data from the field info metadata or empty if not present.
    """
    for item in info.metadata:
        if isinstance(item, GlotaranFieldMetadata):
            return item
    return GlotaranFieldMetadata()


class ItemAttribute(FieldInfo):  # type:ignore[misc]
    """An attribute for items.

    A thin wrapper around pydantic.fields.FieldInfo.
    """

    def __init__(
        self,
        *,
        description: str,
        default: Any = PydanticUndefined,  # noqa: ANN401
        factory: Callable[[], Any] | None = None,
        validator: Callable | None = None,
    ) -> None:
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
        glotaran_field_metadata = GlotaranFieldMetadata()
        if validator is not None:
            glotaran_field_metadata[META_VALIDATOR] = validator
        if factory is not None:
            super().__init__(default_factory=factory, description=description)
        else:
            super().__init__(default=default, description=description)
        self.metadata.append(glotaran_field_metadata)


def Attribute(  # noqa: N802
    *,
    description: str,
    default: Any = PydanticUndefined,  # noqa: ANN401
    factory: Callable[[], Any] | None = None,
    validator: Callable | None = None,
) -> Any:  # noqa: ANN401
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", protected_namespaces=()
    )


class TypedItem(Item):
    """An item with a type."""

    type: Literal[None]
    __item_types__: ClassVar[list[type[Item]]]  # type:ignore[valid-type]

    def __init_subclass__(cls) -> None:
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
        return Annotated[Union[tuple(cls.__item_types__)], Field(discriminator="type")]  # noqa: UP007


@cache
def get_structure_and_type_from_field(
    info: FieldInfo,
) -> tuple[None | list | dict, type]:
    """Get the structure and type from a field.

    Parameters
    ----------
    info: FieldInfo
        The field.

    Returns
    -------
    tuple[None | list | dict, type]:
        The structure and type as atuple.
    """
    definition = strip_option_type_from_definition(info.annotation)  # type:ignore[arg-type]
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


def strip_structure_type_from_definition(
    definition: type,
) -> tuple[None | list | dict, type]:
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
    item: type[ItemT] | ItemT, field_type: type
) -> Generator[tuple[str, FieldInfo]]:
    """Iterate over all fields of the given types.

    Parameters
    ----------
    item: type[Item]
        The item type.
    field_type: type
        The field type.

    Yields
    ------
    tuple[str, FieldInfo]
        The matching attributes.
    """
    for name, info in item.model_fields.items():
        _, item_type = get_structure_and_type_from_field(info)
        with contextlib.suppress(TypeError):
            # issubclass does for some reason not work with e.g. tuple as item_type
            # and Parameter as attr_type
            if (
                hasattr(item_type, "__origin__")
                and issubclass(
                    typing.get_origin(item_type),  # type:ignore[arg-type]
                    typing.Annotated,  # type:ignore[arg-type]
                )
                and typing.get_origin(typing.get_args(item_type)[0]) is typing.Union
            ):
                item_type = typing.get_args(typing.get_args(item_type)[0])[0]
            if isclass(item_type) and issubclass(item_type, field_type):
                yield name, info


def iterate_item_fields(
    item: type[ItemT] | ItemT,
) -> Generator[tuple[str, FieldInfo]]:
    """Iterate over all item fields.

    Parameters
    ----------
    item: type[Item]
        The item type.

    Yields
    ------
    tuple[str, FieldInfo]
        The item fields.
    """
    yield from iterate_fields_of_type(item, Item)


def iterate_parameter_fields(
    item: type[ItemT] | ItemT,
) -> Generator[tuple[str, FieldInfo]]:
    """Iterate over all parameter fields.

    Parameters
    ----------
    item: type[Item]
        The item type.

    Yields
    ------
    tuple[str, FieldInfo]
        The parameter fields.
    """
    yield from iterate_fields_of_type(item, Parameter)


def add_to_initial(label: str, parameters: Parameters, initial: Parameters) -> Parameter:
    if not parameters.has(label):
        for dep_label in initial.get(label).get_dependency_parameters():
            add_to_initial(dep_label, parameters, initial)
        parameters.add(initial.get(label).model_copy())
    return parameters.get(label)


def resolve_parameter(
    parameter: Parameter | float | str, parameters: Parameters, initial: Parameters
) -> Parameter | float:
    if isinstance(parameter, str):
        parameter = add_to_initial(parameter, parameters, initial)
    return parameter


def resolve_item_parameters(  # noqa: C901
    item: ItemT, parameters: Parameters, initial: Parameters | None = None
) -> ItemT:
    resolved: dict[str, Any] = {}
    initial = initial or parameters

    for name, info in iterate_parameter_fields(item):
        value = getattr(item, name)
        if value is None:
            continue
        structure, _ = get_structure_and_type_from_field(info)
        if structure is None:
            resolved[name] = resolve_parameter(value, parameters, initial)
        elif structure is list:
            resolved[name] = [resolve_parameter(v, parameters, initial) for v in value]
        elif structure is dict:
            resolved[name] = {
                k: resolve_parameter(v, parameters, initial) for k, v in value.items()
            }

    for name, info in iterate_item_fields(item):
        value = getattr(item, name)
        if value is None:
            continue
        structure, item_type = get_structure_and_type_from_field(info)
        if structure is None:
            resolved[name] = resolve_item_parameters(value, parameters, initial)
        elif structure is list:
            resolved[name] = [resolve_item_parameters(v, parameters, initial) for v in value]
        elif structure is dict:
            resolved[name] = {
                k: resolve_item_parameters(v, parameters, initial) for k, v in value.items()
            }
    return item.model_copy(update=resolved)


def get_item_issues(item: Item, parameters: Parameters) -> list[ItemIssue]:
    issues = []
    for name, info in iterate_item_fields(item):
        value = getattr(item, name)
        if value is None:
            continue

        glotaran_field_metadata = extract_glotaran_field_metadata(info)
        if glotaran_field_metadata.validator is not None:
            issues += glotaran_field_metadata.validator(value, item, parameters)
        structure, _ = get_structure_and_type_from_field(info)
        if structure is None:
            issues += get_item_issues(value, parameters)
        else:
            values = value.values() if structure is dict else value
            for v in values:
                issues += get_item_issues(v, parameters)

    for name, info in iterate_parameter_fields(item):
        value = getattr(item, name)
        if value is None:
            continue
        structure, _ = get_structure_and_type_from_field(info)
        if structure is None:
            if isinstance(value, str) and not parameters.has(value):
                issues += [ParameterIssue(value)]
        else:
            values = value.values() if structure is dict else value
            issues += [
                ParameterIssue(v) for v in values if isinstance(v, str) and not parameters.has(v)
            ]

    return issues
