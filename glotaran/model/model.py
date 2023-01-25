"""This module contains the model."""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from uuid import uuid4

from attr import asdict
from attr import fields
from attr import ib
from attrs import Attribute
from attrs import define
from attrs import make_class
from attrs import resolve_types

from glotaran.io import load_model
from glotaran.model.clp_constraint import ClpConstraint
from glotaran.model.clp_penalties import ClpPenalty
from glotaran.model.clp_relation import ClpRelation
from glotaran.model.dataset_group import DatasetGroup
from glotaran.model.dataset_group import DatasetGroupModel
from glotaran.model.dataset_model import DatasetModel
from glotaran.model.item import Item
from glotaran.model.item import ItemIssue
from glotaran.model.item import ModelItem
from glotaran.model.item import TypedItem
from glotaran.model.item import get_item_issues
from glotaran.model.item import item_to_markdown
from glotaran.model.item import iterate_parameter_names_and_labels
from glotaran.model.item import model_attributes
from glotaran.model.item import strip_type_and_structure_from_attribute
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters
from glotaran.utils.ipython import MarkdownStr

DEFAULT_DATASET_GROUP = "default"
META_ITEMS = "__glotaran_items__"
META = {META_ITEMS: True}


class ModelError(Exception):
    """Raised when a model contains errors."""

    def __init__(self, error: str):
        """Create a model error.

        Parameters
        ----------
        error: str
            The error string.
        """
        super().__init__(f"ModelError: {error}")


def _load_item_from_dict(
    item_type: type[Item], value: Item | Mapping, extra: dict[str, Any] | None = None
) -> Item:
    """Load an item from a dictionary.

    Parameters
    ----------
    item_type: type[Item]
        The item type.
    value: Item | dict
        The value to load from.
    extra: dict[str, Any] | None
        Extra arguments for the item.

    Returns
    -------
    Item

    Raises
    ------
    ModelError
        Raised if a modelitem is missing.
    """
    if not isinstance(value, Item):
        if extra:
            value = value | extra
        if issubclass(item_type, TypedItem):
            try:
                item_type = item_type.get_item_type_class(value["type"])
            except KeyError:
                raise ModelError(f"Missing 'type' for item {item_type}")
        return item_type(**(value))
    return value


def _load_model_items_from_dict(
    item_type: type[Item], item_dict: Mapping[str, ModelItem | dict]
) -> dict[str, ModelItem]:
    """Load a model items from a dictionary.

    Parameters
    ----------
    item_type: type[Item]
        The item type.
    item_dict: dict[str, ModelItem | dict]
        The item dictionary.

    Returns
    -------
    dict[str, ModelItem]
    """
    return {
        label: _load_item_from_dict(item_type, value, extra={"label": label})  # type:ignore[misc]
        for label, value in item_dict.items()
    }


def _load_global_items_from_dict(
    item_type: type[Item], item_list: list[Item | dict]
) -> list[Item]:
    """Load an item from a dictionary.

    Parameters
    ----------
    item_type: type[Item]
        The item type.
    item_list: list[Item | dict]
        The list of item dicts.

    Returns
    -------
    list[Item]
    """
    return [_load_item_from_dict(item_type, value) for value in item_list]


def _load_dataset_groups(
    dataset_groups: dict[str, DatasetGroupModel | Any]
) -> dict[str, DatasetGroupModel]:
    """Add the default dataset group if not present.

    Parameters
    ----------
    dataset_groups: dict[str, DatasetGroupModel]
        The dataset groups.

    Returns
    -------
    dict[str, DatasetGroupModel]
    """
    dataset_group_items = _load_model_items_from_dict(DatasetGroupModel, dataset_groups)
    if DEFAULT_DATASET_GROUP not in dataset_group_items:
        dataset_group_items[DEFAULT_DATASET_GROUP] = DatasetGroupModel(  # type:ignore[call-arg]
            label=DEFAULT_DATASET_GROUP
        )
    return dataset_group_items  # type:ignore[return-value]


def _global_item_attribute(item_type: type[Item]) -> Attribute:
    """Create a global item attribute.

    Parameters
    ----------
    item_type: type[Item]
        The item type.

    Returns
    -------
    Attribute
    """
    return ib(
        factory=list,
        converter=lambda value: _load_global_items_from_dict(item_type, value),
        metadata=META,
    )


def _model_item_attribute(item_type: type[ModelItem]):
    """Create a model item attribute.

    Parameters
    ----------
    item_type: type[ModelItem]
        The item type.

    Returns
    -------
    Attribute
    """
    return ib(
        type=dict[str, item_type],  # type:ignore[valid-type]
        factory=dict,
        converter=lambda value: _load_model_items_from_dict(item_type, value),
        metadata=META,
    )


def _create_attributes_for_item(item_type: type[Item]) -> dict[str, Attribute]:
    """Create attributes for an item.

    Parameters
    ----------
    item_type: type[Item]
        The item type.

    Returns
    -------
    dict[str, Attribute]
    """
    attributes = {}
    for model_item in model_attributes(item_type, with_alias=False):
        _, model_item_type = strip_type_and_structure_from_attribute(model_item)
        attributes[model_item.name] = _model_item_attribute(model_item_type)
    return attributes


@define(kw_only=True)
class Model:
    """A model for global target analysis."""

    loader: ClassVar[Callable] = load_model

    source_path: str | None = ib(default=None, init=False, repr=False)
    clp_penalties: list[ClpPenalty] = _global_item_attribute(ClpPenalty)
    clp_constraints: list[ClpConstraint] = _global_item_attribute(ClpConstraint)
    clp_relations: list[ClpRelation] = _global_item_attribute(ClpRelation)

    dataset_groups: dict[str, DatasetGroupModel] = ib(
        factory=dict, converter=_load_dataset_groups, metadata=META
    )

    dataset: dict[str, DatasetModel]

    megacomplex: dict[str, Megacomplex] = ib(
        factory=dict,
        converter=lambda value: _load_model_items_from_dict(Megacomplex, value),
        metadata=META,
    )

    weights: list[Weight] = _global_item_attribute(Weight)

    @classmethod
    def create_class(cls, attributes: dict[str, Attribute]) -> type[Model]:
        """Create model class.

        Parameters
        ----------
        attributes: dict[str, Attribute]
            The model attributes.

        Returns
        -------
        type[Model]
        """
        cls_name = f"GlotaranModel_{str(uuid4()).replace('-','_')}"
        return make_class(cls_name, attributes, bases=(cls,))

    @classmethod
    def create_class_from_megacomplexes(
        cls, megacomplexes: Iterable[type[Megacomplex]]
    ) -> type[Model]:
        """Create model class for megacomplexes.

        Parameters
        ----------
        megacomplexes: list[type[Megacomplex]]
            The megacomplexes.

        Returns
        -------
        type[Model]
        """
        attributes: dict[str, Attribute] = {}
        dataset_types = set()
        for megacomplex in megacomplexes:
            if dataset_model_type := megacomplex.get_dataset_model_type():
                dataset_types |= {
                    dataset_model_type,
                }
            attributes |= _create_attributes_for_item(megacomplex)

        dataset_type = (
            DatasetModel
            if len(dataset_types) == 0
            else make_class(
                f"GlotaranDataset_{str(uuid4()).replace('-','_')}",
                [],
                bases=tuple(dataset_types),
                collect_by_mro=True,
            )
        )
        resolve_types(dataset_type)

        attributes.update(_create_attributes_for_item(dataset_type))

        attributes["dataset"] = _model_item_attribute(dataset_type)

        return cls.create_class(attributes)

    def as_dict(self) -> dict:
        """Get the model as dictionary.

        Returns
        -------
        dict
        """
        return asdict(
            self,
            recurse=True,
            retain_collection_types=True,
            filter=lambda attr, _: attr.name != "source_path",
        )

    def get_dataset_groups(self) -> dict[str, DatasetGroup]:
        """Get the dataset groups.

        Returns
        -------
        dict[str, DatasetGroup]

        Raises
        ------
        ModelError
            Raised if a dataset group is unknown.
        """
        groups = {}
        for dataset_model in self.dataset.values():
            group = dataset_model.group
            if group not in groups:
                try:
                    group_model = self.dataset_groups[group]
                except KeyError:
                    raise ModelError(f"Unknown dataset group '{group}'")
                groups[group] = DatasetGroup(  # type:ignore[call-arg]
                    residual_function=group_model.residual_function,
                    link_clp=group_model.link_clp,
                    model=self,
                )
            groups[group].dataset_models[dataset_model.label] = dataset_model
        return groups

    def iterate_items(self) -> Generator[tuple[str, dict[str, Item] | list[Item]], None, None]:
        """Iterate items.

        Yields
        ------
        tuple[str, dict[str, Item] | list[Item]]
            The name of the item and the individual items of the type.
        """
        for attr in fields(self.__class__):
            if META_ITEMS in attr.metadata:
                yield attr.name, getattr(self, attr.name)

    def iterate_all_items(self) -> Generator[Item, None, None]:
        """Iterate the individual items.

        Yields
        ------
        Item
            The individual item.
        """
        for _, items in self.iterate_items():
            yield from items.values() if isinstance(items, dict) else items

    def get_parameter_labels(self) -> set[str]:
        """Get all parameter labels.

        Returns
        -------
        set[str]
        """
        return {
            label
            for item in self.iterate_all_items()
            for _, label in iterate_parameter_names_and_labels(item)
        }

    def generate_parameters(self) -> Parameters:
        """Generate parameters for the model.

        Returns
        -------
        Parameters
            The generated parameters.

        .. # noqa: D414
        """
        return Parameters(
            {
                label: Parameter(label=label, value=0)  # type:ignore[call-arg]
                for label in self.get_parameter_labels()
            }
        )

    def get_issues(self, *, parameters: Parameters | None = None) -> list[ItemIssue]:
        """Get issues.

        Parameters
        ----------
        parameters: Parameters | None
            The parameters.

        Returns
        -------
        list[ItemIssue]
        """
        issues = []
        for item in self.iterate_all_items():
            issues += get_item_issues(item=item, model=self, parameters=parameters)
        return issues

    def validate(
        self, parameters: Parameters | None = None, raise_exception: bool = False
    ) -> MarkdownStr:
        """Get a string listing all issues in the model and missing parameters if specified.

        Parameters
        ----------
        parameters: Parameters | None
            The parameters.
        raise_exception: bool
            Whether to raise an exception on failed validation.

        Returns
        -------
        MarkdownStr

        Raises
        ------
        ModelError
            Raised if validation fails and raise_exception is true.
        """
        result = ""

        if issues := self.get_issues(parameters=parameters):
            result = f"Your model has {len(issues)} problem{'s' if len(issues) > 1 else ''}:\n"
            for issue in issues:
                result += f"\n * {issue.to_string()}"
            if raise_exception:
                raise ModelError(result)
        else:
            result = "Your model is valid."
        return MarkdownStr(result)

    def valid(self, parameters: Parameters | None = None) -> bool:
        """Check if the model is valid.

        Parameters
        ----------
        parameters: Parameters | None
            The parameters.

        Returns
        -------
        bool
        """
        return len(self.get_issues(parameters=parameters)) == 0

    def markdown(
        self,
        parameters: Parameters | None = None,
        initial_parameters: Parameters | None = None,
        base_heading_level: int = 1,
    ) -> MarkdownStr:
        """Format the model as Markdown string.

        Parameters will be included if specified.

        Parameters
        ----------
        parameters: Parameters | None
            Parameter to include.
        initial_parameters: Parameters | None
            Initial values for the parameters.
        base_heading_level: int
            Base heading level of the markdown sections.

            E.g.:

            - If it is 1 the string will start with '# Model'.
            - If it is 3 the string will start with '### Model'.

        Returns
        -------
        MarkdownStr
        """
        base_heading = "#" * base_heading_level
        string = f"{base_heading} Model\n\n"

        for name, items in self.iterate_items():
            if not items:
                continue

            string += f"{base_heading}# {name.replace('_', ' ').title()}\n\n"

            if isinstance(items, dict):
                items = items.values()  # type:ignore[assignment]
            for item in items:
                assert isinstance(item, Item)
                item_str = item_to_markdown(
                    item, parameters=parameters, initial_parameters=initial_parameters
                ).split("\n")
                string += f"- **{getattr(item, 'label', '&nbsp;')}**\n"
                for s in item_str[1:]:
                    string += f"{s}\n"
            string += "\n"
        return MarkdownStr(string)

    def _repr_markdown_(self) -> str:
        """Render ``ipython`` markdown.

        Returns
        -------
        str
        """
        return str(self.markdown(base_heading_level=3))
