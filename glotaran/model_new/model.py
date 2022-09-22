from __future__ import annotations

from typing import Generator
from uuid import uuid4

from attr import asdict
from attr import fields
from attr import ib
from attrs import Attribute
from attrs import define
from attrs import field
from attrs import make_class
from attrs import resolve_types

from glotaran.model_new.dataset_group import DatasetGroupModel
from glotaran.model_new.dataset_model import DatasetModel
from glotaran.model_new.item import Item
from glotaran.model_new.item import ItemIssue
from glotaran.model_new.item import ModelItemTyped
from glotaran.model_new.item import get_item_issues
from glotaran.model_new.item import item_to_markdown
from glotaran.model_new.item import iterate_parameter_names_and_labels
from glotaran.model_new.item import model_attributes
from glotaran.model_new.item import strip_type_and_structure_from_attribute
from glotaran.model_new.megacomplex import Megacomplex
from glotaran.model_new.weight import Weight
from glotaran.parameter import ParameterGroup
from glotaran.utils.ipython import MarkdownStr

DEFAULT_DATASET_GROUP = "default"
META_ITEMS = "__glotaran_items__"
META = {META_ITEMS: True}


class ModelError(Exception):
    """Raised when a model contains errors."""

    def __init__(self, error: str):
        super().__init__(f"ModelError: {error}")


def _load_item_from_dict(cls, value: any, extra: dict[str, any] = {}) -> any:
    if isinstance(value, dict):
        if issubclass(cls, ModelItemTyped):
            item_type = value["type"]
            cls = cls.get_item_type_class(item_type)
        value = cls(**(value | extra))
    return value


def _load_model_items_from_dict(cls, item_dict: dict[str, any]) -> dict[str, any]:
    return {
        label: _load_item_from_dict(cls, value, extra={"label": label})
        for label, value in item_dict.items()
    }


def _load_global_items_from_dict(cls, item_list: list[any]) -> list[any]:
    return [_load_item_from_dict(cls, value) for value in item_list]


def _add_default_dataset_group(
    dataset_groups: dict[str, DatasetGroupModel]
) -> dict[str, DatasetGroupModel]:
    dataset_groups = _load_model_items_from_dict(DatasetGroupModel, dataset_groups)
    if DEFAULT_DATASET_GROUP not in dataset_groups:
        dataset_groups[DEFAULT_DATASET_GROUP] = DatasetGroupModel(label=DEFAULT_DATASET_GROUP)
    return dataset_groups


def _model_item_attribute(model_item_type: type):
    return ib(
        type=dict[str, model_item_type],
        factory=dict,
        converter=lambda value: _load_model_items_from_dict(model_item_type, value),
        metadata=META,
    )


def _create_attributes_for_item(item: Item) -> dict[str, Attribute]:
    attributes = {}
    for model_item in model_attributes(item, with_alias=False):
        _, model_item_type = strip_type_and_structure_from_attribute(model_item)
        attributes[model_item.name] = _model_item_attribute(model_item_type)
    return attributes


@define(kw_only=True)
class Model:

    dataset_groups: dict[str, DatasetGroupModel] = field(
        factory=dict, converter=_add_default_dataset_group, metadata=META
    )

    dataset: dict[str, DatasetModel]

    megacomplex: dict[str, Megacomplex] = field(
        factory=dict,
        converter=lambda value: _load_model_items_from_dict(Megacomplex, value),
        metadata=META,
    )

    weights: list[Weight] = field(
        factory=list,
        converter=lambda value: _load_global_items_from_dict(Weight, value),
        metadata=META,
    )

    @classmethod
    def create_class(cls, attributes: dict[str, Attribute]) -> Model:
        cls_name = f"GlotaranModel_{str(uuid4()).replace('-','_')}"
        return make_class(cls_name, attributes, bases=(cls,))

    @classmethod
    def create_class_from_megacomplexes(cls, megacomplexes: list[Megacomplex]) -> Model:
        attributes: dict[str, Attribute] = {}
        dataset_types = set()
        for megacomplex in megacomplexes:
            if dataset_model_type := megacomplex.get_dataset_model_type():
                dataset_types |= {
                    dataset_model_type,
                }
            attributes.update(_create_attributes_for_item(megacomplex))

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
        return asdict(self, recurse=True, retain_collection_types=True)

    def iterate_items(self) -> Generator[tuple[dict[str, Item] | list[Item]], None, None]:
        for attr in fields(self.__class__):
            if META_ITEMS in attr.metadata:
                yield attr.name, getattr(self, attr.name)

    def iterate_all_items(self) -> Generator[Item, None, None]:
        for _, items in self.iterate_items():
            iter = items.values() if isinstance(items, dict) else items
            yield from iter

    def get_parameter_labels(self) -> set[str]:
        return {
            label
            for item in self.iterate_all_items()
            for _, label in iterate_parameter_names_and_labels(item)
        }

    def get_issues(self, *, parameters: ParameterGroup | None = None) -> list[ItemIssue]:
        issues = []
        for item in self.iterate_all_items():
            issues += get_item_issues(item=item, model=self, parameters=parameters)
        return issues

    def validate(
        self, parameters: ParameterGroup = None, raise_exception: bool = False
    ) -> MarkdownStr:
        """
        Returns a string listing all issues in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        result = ""

        if issues := self.get_issues(parameters):
            result = f"Your model has {len(issues)} problem{'s' if len(issues) > 1 else ''}:\n"
            for p in issues:
                result += f"\n * {p}"
            if raise_exception:
                raise ModelError(issues)
        else:
            result = "Your model is valid."
        return MarkdownStr(result)

    def valid(self, parameters: ParameterGroup = None) -> bool:
        """Returns `True` if the number problems in the model is 0, else `False`

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        return len(self.get_issues(parameters=parameters)) == 0

    def markdown(
        self,
        parameters: ParameterGroup = None,
        initial_parameters: ParameterGroup = None,
        base_heading_level: int = 1,
    ) -> MarkdownStr:
        """Formats the model as Markdown string.

        Parameters will be included if specified.

        Parameters
        ----------
        parameter: ParameterGroup
            Parameter to include.
        initial_parameters: ParameterGroup
            Initial values for the parameters.
        base_heading_level: int
            Base heading level of the markdown sections.

            E.g.:

            - If it is 1 the string will start with '# Model'.
            - If it is 3 the string will start with '### Model'.
        """
        base_heading = "#" * base_heading_level
        string = f"{base_heading} Model\n\n"

        for name, items in self.iterate_items():
            if not items:
                continue

            string += f"{base_heading}# {name.replace('_', ' ').title()}\n\n"

            if isinstance(items, dict):
                items = items.values()
            for item in items:
                item_str = item_to_markdown(
                    item, parameters=parameters, initial_parameters=initial_parameters
                ).split("\n")
                string += f"* {item_str[0]}\n"
                for s in item_str[1:]:
                    string += f"  {s}\n"
            string += "\n"
        return MarkdownStr(string)

    def _repr_markdown_(self) -> str:
        """Special method used by ``ipython`` to render markdown."""
        return str(self.markdown(base_heading_level=3))
