"""This module contains the items."""
from __future__ import annotations

import contextlib
from inspect import getmro
from inspect import isclass
from textwrap import indent
from types import NoneType
from types import UnionType
from typing import TYPE_CHECKING
from typing import Any
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
from attrs import evolve
from attrs import field
from attrs import fields
from attrs import resolve_types

from glotaran.parameter import Parameter
from glotaran.parameter import Parameters
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from glotaran.model.model import Model


META_ALIAS = "__glotaran_alias__"
META_VALIDATOR = "__glotaran_validator__"


class ItemIssue:
    """Baseclass for item issues."""

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError

    def __rep__(self) -> str:
        """Get the representation."""
        return self.to_string()


class ModelItemIssue(ItemIssue):
    """Issue for missing model items."""

    def __init__(self, item_name: str, label: str):
        """Create a model issue.

        Parameters
        ----------
        item_name : str
            The name of the item.
        label : str
            The item label.
        """
        self._item_name = item_name
        self._label = label

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return f"Missing model item '{self._item_name}' with label '{self._label}'."


class ParameterIssue(ItemIssue):
    """Issue for missing parameters."""

    def __init__(self, label: str):
        """Create a parameter issue.

        Parameters
        ----------
        label : str
            The parameter label.
        """
        self._label = label

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return f"Missing parameter with label '{self._label}'."


@define(kw_only=True, slots=False)
class Item:
    """A baseclass for items."""


@define(kw_only=True, slots=False)
class ModelItem(Item):
    """An item with a label."""

    label: str


@define(kw_only=True, slots=False)
class TypedItem(Item):
    """An item with a type."""

    type: str
    __item_types__: ClassVar[dict[str, Type]]

    @classmethod
    def _register_item_class(cls):
        """Register a class as type."""
        item_type = cls.get_item_type()
        if item_type is not NOTHING:
            cls.__item_types__[item_type] = cls

    @classmethod
    def get_item_type(cls) -> str:
        """Get the type string.

        Returns
        -------
        str
        """
        return fields(cls).type.default

    @classmethod
    def get_item_types(cls) -> list[str]:
        """Get all type strings.

        Returns
        -------
        list[str]
        """
        return list(cls.__item_types__.keys())

    @classmethod
    def get_item_type_class(cls, item_type: str) -> Type:
        """Get the type for a type string.

        Parameters
        ----------
        item_type: str
            The type string.
        Returns
        -------
        Type
        """
        return cls.__item_types__[item_type]


@define(kw_only=True, slots=False)
class ModelItemTyped(TypedItem, ModelItem):
    """A model item with a type."""


ItemT = TypeVar("ItemT", bound="Item")
ModelItemT = TypeVar("ModelItemT", bound="ModelItem")

ParameterType: TypeAlias = Parameter | str
ModelItemType: TypeAlias = ModelItemT | str


def item_to_markdown(
    item: Item, parameters: Parameters | None = None, initial_parameters: Parameters | None = None
) -> MarkdownStr:
    """Get the item as markdown string.

    Parameters
    ----------
    item: Item
        The item.
    parameters: Parameters | None
        The parameters.
    initial_parameters: Parameters | None
        The initial parameters.

    Returns
    -------
    MarkdownStr
    """
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
                    k: parameters.get(v.label if isinstance(v, Parameter) else v).markdown(
                        parameters, initial_parameters
                    )
                    for k, v in value.items()
                }
            elif structure is list:
                value = [
                    parameters.get(v.label if isinstance(v, Parameter) else v).markdown(
                        parameters, initial_parameters
                    )
                    for v in value
                ]
            else:
                value = parameters.get(
                    value.label if isinstance(value, Parameter) else value
                ).markdown(parameters, initial_parameters)

        property_md = indent(f"- _{name.replace('_', ' ').title()}_: {value}\n", "  ")

        md += property_md

    return MarkdownStr(md)


def iterate_attributes_of_type(
    item: type[Item], attr_type: type
) -> Generator[Attribute, None, None]:
    """Get attributes of type from an item type.

    Parameters
    ----------
    item: type[Item]
        The item type.
    attr_type: type
        The attribute type.

    Yields
    ------
    Attribute
        The attributes.
    """
    for attr in fields(item):
        _, item_type = strip_type_and_structure_from_attribute(attr)
        with contextlib.suppress(TypeError):
            # issubclass does for some reason not work with e.g. tuple as item_type
            # and Parameter as attr_type
            if isclass(item_type) and issubclass(item_type, attr_type):
                yield attr


def model_attributes(
    item: type[Item], with_alias: bool = True
) -> Generator[Attribute, None, None]:
    """Get model attributes from an item type.

    Parameters
    ----------
    item: type[Item]
        The item type.
    with_alias: bool
        Whether to return aliased attributes.

    Yields
    ------
    Attribute
        The model attributes.
    """
    for attr in iterate_attributes_of_type(item, ModelItem):
        if with_alias or META_ALIAS not in attr.metadata:
            yield attr


def parameter_attributes(item: type[Item]) -> Generator[Attribute, None, None]:
    """Get parameter attributes from an item type.

    Parameters
    ----------
    item: type[Item]
        The item type.

    Yields
    ------
    Attribute
        The parameter attributes.
    """
    yield from iterate_attributes_of_type(item, Parameter)


def iterate_names_and_labels(
    item: Item, attributes: Generator[Attribute, None, None]
) -> Generator[tuple[str, str], None, None]:
    """Get attribute names and labels.

    Parameters
    ----------
    item: Item
        The item.
    attributes: Generator[Attribute, None, None]
        The attributes.

    Yields
    ------
    tuple[str, str]
        The name and the label.
    """
    for attr in attributes:
        structure, _ = strip_type_and_structure_from_attribute(attr)
        value = getattr(item, attr.name)
        name: str = attr.metadata.get(META_ALIAS, attr.name)

        if not value:
            continue

        if structure is dict:
            for v in value.values():
                yield name, v if isinstance(v, str) else (name, v.label)  # type:ignore[misc]

        elif structure is list:
            for v in value:
                yield name, v if isinstance(v, str) else (name, v.label)  # type:ignore[misc]

        else:
            yield name, value if isinstance(value, str) else (
                name,
                value.label,  # type:ignore[misc]
            )


def iterate_model_item_names_and_labels(item: Item) -> Generator[tuple[str, str], None, None]:
    """Get model item names and labels.

    Parameters
    ----------
    item: Item
        The item.

    Yields
    ------
    tuple[str, str]
        The name and the label.
    """
    yield from iterate_names_and_labels(item, model_attributes(item.__class__))


def iterate_parameter_names_and_labels(item: Item) -> Generator[tuple[str, str], None, None]:
    """Get parameter item names and labels.

    Parameters
    ----------
    item: Item
        The item.

    Yields
    ------
    tuple[str, str]
        The name and the label.
    """
    yield from iterate_names_and_labels(item, parameter_attributes(item.__class__))


def fill_item_attributes(
    item: Item,
    iterator: Iterator[Attribute],
    fill_function: Callable[[str, str], Parameter | ModelItem],
):
    """Fill item attributes.

    Parameters
    ----------
    item: Item
        The item.
    iterator: Iterator[Attribute]
        An iterator over attributes.
    fill_function: Callable[[str, str], Parameter | ModelItem]
        The function to fill the values.
    """
    for attr in iterator:
        value = getattr(item, attr.name)
        if not value:
            continue

        structure, _ = strip_type_and_structure_from_attribute(attr)
        name = attr.metadata.get(META_ALIAS, attr.name)
        if structure is dict:
            value = {
                k: fill_function(name, v) if isinstance(v, str) else fill_function(name, v.label)
                for k, v in value.items()
            }
        elif structure is list:
            value = [
                fill_function(name, v) if isinstance(v, str) else fill_function(name, v.label)
                for v in value
            ]
        else:
            value = (
                fill_function(name, value)
                if isinstance(value, str)
                else fill_function(name, value.label)
            )

        setattr(item, attr.name, value)


def fill_item(item: ItemT, model: Model, parameters: Parameters) -> ItemT:
    """Fill an item.

    Parameters
    ----------
    item: ItemT
        The item.
    model: Model
        The model.
    parameters: Parameters
        The parameters.

    Returns
    -------
    ItemT
        The filled item.
    """
    item = evolve(item)
    fill_item_model_attributes(item, model, parameters)
    fill_item_parameter_attributes(item, parameters)
    return item


def fill_item_model_attributes(item: Item, model: Model, parameters: Parameters):
    """Fill item model attributes.

    Parameters
    ----------
    item: Item
        The item.
    model: Model
        The model.
    parameters: Parameters
        The parameters.
    """
    fill_item_attributes(
        item,
        model_attributes(item.__class__),
        lambda name, label: fill_item(getattr(model, name)[label], model, parameters),
    )


def fill_item_parameter_attributes(item: Item, parameters: Parameters):
    """Fill item parameter attributes.

    Parameters
    ----------
    item: Item
        The item.
    parameters: Parameters
        The parameters.
    """
    fill_item_attributes(
        item, parameter_attributes(item.__class__), lambda _, label: parameters.get(label)
    )


def get_item_model_issues(item: Item, model: Model) -> list[ItemIssue]:
    """Get model item issues for an item.

    Parameters
    ----------
    item: Item
        The item.
    model: Model
        The model.

    Returns
    -------
    list[ItemIssue]
    """
    return [
        ModelItemIssue(name, label)
        for name, label in iterate_model_item_names_and_labels(item)
        if label not in getattr(model, name)
    ]


def get_item_parameter_issues(item: Item, parameters: Parameters) -> list[ItemIssue]:
    """Get model item issues for an item.

    Parameters
    ----------
    item: Item
        The item.
    parameters: Parameters
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return [
        ParameterIssue(label)
        for name, label in iterate_parameter_names_and_labels(item)
        if not parameters.has(label)
    ]


def get_item_validator_issues(
    item: Item, model: Model, parameters: Parameters | None = None
) -> list[ItemIssue]:
    """Get validator issues for an item.

    Parameters
    ----------
    item: Item
        The item.
    model: Model
        The model.
    parameters: Parameters | None
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    issues = []
    for name, validator in [
        (attr.name, attr.metadata[META_VALIDATOR])
        for attr in fields(item.__class__)
        if META_VALIDATOR in attr.metadata
    ]:
        issues += validator(getattr(item, name), item, model, parameters)

    return issues


def get_item_issues(
    *, item: Item, model: Model, parameters: Parameters | None = None
) -> list[ItemIssue]:
    """Get issues for an item.

    Parameters
    ----------
    item: Item
        The item.
    model: Model
        The model.
    parameters: Parameters | None
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    issues = get_item_model_issues(item, model)
    issues += get_item_validator_issues(item, model, parameters)
    if parameters is not None:
        issues += get_item_parameter_issues(item, parameters)
    return issues


def strip_type_and_structure_from_attribute(attr: Attribute) -> tuple[None | list | dict, type]:
    """Strip the type and the structure from an attribute.

    Parameters
    ----------
    attr: Attribute
        The attribute.

    Returns
    -------
    tuple[None | list | dict, type]:
        The structure and the type.
    """
    definition = attr.type
    definition = strip_option_type(definition)
    structure, definition = strip_structure_type(definition)
    definition = strip_option_type(definition, strip_type=str)
    return structure, definition


def strip_option_type(definition: type, strip_type: type = NoneType) -> type:
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


def strip_structure_type(definition: type) -> tuple[None | list | dict, type]:
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


def item(cls: type[ItemT]) -> type[ItemT]:
    """Create an item from a class.

    Parameters
    ----------
    cls: type[ItemT]
        The class.

    Returns
    -------
    type[ItemT]
    """
    parent = getmro(cls)[1]
    cls = define(kw_only=True, slots=False)(cls)
    if parent in (TypedItem, ModelItemTyped):
        assert issubclass(cls, TypedItem)
        cls.__item_types__ = {}
    elif issubclass(cls, TypedItem):
        cls._register_item_class()
    resolve_types(cls)
    return cls


def attribute(
    *,
    alias: str | None = None,
    default: Any = NOTHING,
    factory: Callable[[], Any] | None = None,
    validator: Callable[[Any, Item, Model, Parameters | None], list[ItemIssue]] | None = None,
) -> Attribute:
    """Create an attribute for an item.

    Parameters
    ----------
    alias: str | None
        The alias of the attribute (only useful for model items).
    default: Any
        The default value of the attribute.
    factory: Callable[[], Any] | None
        A factory function for the attribute.
    validator: Callable[[Any, Item, Model, Parameters | None], list[ItemIssue]] | None
        A validator function for the attribute.

    Returns
    -------
    Attribute
    """
    metadata: dict[str, Any] = {}
    if alias is not None:
        metadata[META_ALIAS] = alias
    if validator is not None:
        metadata[META_VALIDATOR] = validator
    return field(default=default, factory=factory, metadata=metadata)
