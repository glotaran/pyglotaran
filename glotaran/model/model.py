"""A base class for global analysis models."""
from __future__ import annotations

import copy
from typing import Any
from typing import List
from warnings import warn

import xarray as xr

from glotaran.deprecation import raise_deprecation_error
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.dataset_model import create_dataset_model_type
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import create_model_megacomplex_type
from glotaran.model.relation import Relation
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup
from glotaran.plugin_system.megacomplex_registration import get_megacomplex
from glotaran.utils.ipython import MarkdownStr

default_model_items = {
    "clp_area_penalties": EqualAreaPenalty,
    "clp_constraints": Constraint,
    "clp_relations": Relation,
    "weights": Weight,
}

default_dataset_properties = {
    "megacomplex": List[str],
    "megacomplex_scale": {"type": List[Parameter], "allow_none": True},
    "global_megacomplex": {"type": List[str], "allow_none": True},
    "global_megacomplex_scale": {"type": List[Parameter], "default": None, "allow_none": True},
    "scale": {"type": Parameter, "default": None, "allow_none": True},
}


class Model:
    """A base class for global analysis models."""

    def __init__(
        self,
        *,
        megacomplex_types: dict[str, type[Megacomplex]],
        default_megacomplex_type: str | None = None,
    ):
        self._megacomplex_types = megacomplex_types
        self._default_megacomplex_type = default_megacomplex_type or next(iter(megacomplex_types))

        self._model_items = {}
        self._dataset_properties = {}
        self._add_default_items_and_properties()
        self._add_megacomplexe_types()
        self._add_dataset_type()

    @classmethod
    def from_dict(
        cls,
        model_dict: dict[str, Any],
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = None,
        default_megacomplex_type: str | None = None,
    ) -> Model:
        """Creates a model from a dictionary.

        Parameters
        ----------
        model_dict: dict[str, Any]
            Dictionary containing the model.
        megacomplex_types: dict[str, type[Megacomplex]] | None
            Overwrite 'megacomplex_types' in ``model_dict`` for testing.
        default_megacomplex_type: str | None
            Overwrite 'default-megacomplex' in ``model_dict`` for testing.
        """
        if default_megacomplex_type is None:
            default_megacomplex_type = model_dict.get("default-megacomplex")

        if megacomplex_types is None:
            megacomplex_types = {
                m["type"]: get_megacomplex(m["type"])
                for m in model_dict["megacomplex"].values()
                if "type" in m
            }
        if (
            default_megacomplex_type is not None
            and default_megacomplex_type not in megacomplex_types
        ):
            megacomplex_types[default_megacomplex_type] = get_megacomplex(default_megacomplex_type)
            model_dict.pop("default-megacomplex", None)

        model = cls(
            megacomplex_types=megacomplex_types, default_megacomplex_type=default_megacomplex_type
        )

        model_dict_local = copy.deepcopy(model_dict)  # TODO: maybe redundant?

        # iterate over items
        for name, items in list(model_dict_local.items()):

            if name not in model.model_items:
                warn(f"Unknown model item type '{name}'.")
                continue

            is_list = isinstance(getattr(model, name), list)

            if is_list:
                model._add_list_items(name, items)
            else:
                model._add_dict_items(name, items)

        return model

    def _add_dict_items(self, name: str, items: dict):

        for label, item in items.items():
            item_cls = self.model_items[name]
            is_typed = hasattr(item_cls, "_glotaran_model_item_typed")
            if is_typed:
                if "type" not in item and item_cls.get_default_type() is None:
                    raise ValueError(f"Missing type for attribute '{name}'")
                item_type = item.get("type", item_cls.get_default_type())

                types = item_cls._glotaran_model_item_types
                if item_type not in types:
                    raise ValueError(f"Unknown type '{item_type}' for attribute '{name}'")
                item_cls = types[item_type]
            item["label"] = label
            item = item_cls.from_dict(item)
            getattr(self, name)[label] = item

    def _add_list_items(self, name: str, items: list):

        for item in items:
            item_cls = self.model_items[name]
            is_typed = hasattr(item_cls, "_glotaran_model_item_typed")
            if is_typed:
                if "type" not in item:
                    raise ValueError(f"Missing type for attribute '{name}'")
                item_type = item["type"]

                if item_type not in item_cls._glotaran_model_item_types:
                    raise ValueError(f"Unknown type '{item_type}' for attribute '{name}'")
                item_cls = item_cls._glotaran_model_item_types[item_type]
            item = item_cls.from_dict(item)
            getattr(self, name).append(item)

    def _add_megacomplexe_types(self):

        for name, megacomplex_type in self._megacomplex_types.items():
            if not issubclass(megacomplex_type, Megacomplex):
                raise TypeError(
                    f"Megacomplex type {name}({megacomplex_type}) is not a subclass of Megacomplex"
                )
            self._add_megacomplex_type(megacomplex_type)

        model_megacomplex_type = create_model_megacomplex_type(
            self._megacomplex_types, self.default_megacomplex
        )
        self._add_model_item("megacomplex", model_megacomplex_type)

    def _add_megacomplex_type(self, megacomplex_type: type[Megacomplex]):

        for name, item in megacomplex_type.glotaran_model_items().items():
            self._add_model_item(name, item)

        for name, item in megacomplex_type.glotaran_dataset_model_items().items():
            self._add_model_item(name, item)

        for name, prop in megacomplex_type.glotaran_dataset_properties().items():
            self._add_dataset_property(name, prop)

    def _add_model_item(self, name: str, item: type):
        if name in self._model_items:
            if self.model_items[name] != item:
                raise ModelError(
                    f"Cannot add item of type {name}. Model item '{name}' was already defined"
                    "as a different type."
                )
            return
        self._model_items[name] = item

        if getattr(item, "_glotaran_has_label"):
            setattr(self, f"{name}", {})
        else:
            setattr(self, f"{name}", [])

    def _add_dataset_property(self, name: str, dataset_property: dict[str, any]):
        if name in self._dataset_properties:
            known_type = (
                self._dataset_properties[name]
                if not isinstance(self._dataset_properties, dict)
                else self._dataset_properties[name]["type"]
            )
            new_type = (
                dataset_property
                if not isinstance(dataset_property, dict)
                else dataset_property["type"]
            )
            if known_type != new_type:
                raise ModelError(
                    f"Cannot add dataset property of type {name} as it was already defined"
                    "as a different type."
                )
            return
        self._dataset_properties[name] = dataset_property

    def _add_default_items_and_properties(self):
        for name, item in default_model_items.items():
            self._add_model_item(name, item)

        for name, prop in default_dataset_properties.items():
            self._add_dataset_property(name, prop)

    def _add_dataset_type(self):
        dataset_model_type = create_dataset_model_type(self._dataset_properties)
        self._add_model_item("dataset", dataset_model_type)

    @property
    def model_dimension(self):
        """Deprecated use ``Scheme.model_dimensions['<dataset_name>']`` instead"""
        raise_deprecation_error(
            deprecated_qual_name_usage="Model.model_dimension",
            new_qual_name_usage=("Scheme.model_dimensions['<dataset_name>']"),
            to_be_removed_in_version="0.7.0",
        )

    @property
    def global_dimension(self):
        """Deprecated use ``Scheme.global_dimensions['<dataset_name>']`` instead"""
        raise_deprecation_error(
            deprecated_qual_name_usage="Model.global_dimension",
            new_qual_name_usage=("Scheme.global_dimensions['<dataset_name>']"),
            to_be_removed_in_version="0.7.0",
        )

    @property
    def default_megacomplex(self) -> str:
        """The default megacomplex used by this model."""
        return self._default_megacomplex_type

    @property
    def megacomplex_types(self) -> dict[str, type[Megacomplex]]:
        """The megacomplex types used by this model."""
        return self._megacomplex_types

    @property
    def model_items(self) -> dict[str, type[object]]:
        """The model_items types used by this model."""
        return self._model_items

    @property
    def global_megacomplex(self) -> dict[str, Megacomplex]:
        """Alias for `glotaran.model.megacomplex`. Needed internally."""
        return self.megacomplex

    def as_dict(self) -> dict:
        model_dict = {}
        model_dict["default-megacomplex"] = self.default_megacomplex

        for name in self._model_items:
            items = getattr(self, name)
            if len(items) == 0:
                continue
            if isinstance(items, list):
                model_dict[name] = [item.as_dict() for item in items]
            else:
                model_dict[name] = {label: item.as_dict() for label, item in items.items()}

        return model_dict

    def get_parameters(self) -> list[str]:
        parameters = []
        for item_name in self.model_items:
            items = getattr(self, item_name)
            item_iterator = items if isinstance(items, list) else items.values()
            for item in item_iterator:
                parameters += item.get_parameters()
        return parameters

    def need_index_dependent(self) -> bool:
        """Returns true if e.g. clp_relations with intervals are present."""
        return any(i.interval is not None for i in self.clp_constraints + self.clp_relations)

    def is_groupable(self, parameters: ParameterGroup, data: dict[str, xr.DataArray]) -> bool:
        if any(d.has_global_model() for d in self.dataset.values()):
            return False
        global_dimensions = {
            d.fill(self, parameters).set_data(data[k]).get_global_dimension()
            for k, d in self.dataset.items()
        }
        model_dimensions = {
            d.fill(self, parameters).set_data(data[k]).get_model_dimension()
            for k, d in self.dataset.items()
        }
        return len(global_dimensions) == 1 and len(model_dimensions) == 1

    def problem_list(self, parameters: ParameterGroup = None) -> list[str]:
        """
        Returns a list with all problems in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        problems = []

        for name in self.model_items:
            items = getattr(self, name)
            if isinstance(items, list):
                for item in items:
                    problems += item.validate(self, parameters=parameters)
            else:
                for _, item in items.items():
                    problems += item.validate(self, parameters=parameters)

        return problems

    def validate(self, parameters: ParameterGroup = None, raise_exception: bool = False) -> str:
        """
        Returns a string listing all problems in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        result = ""

        problems = self.problem_list(parameters)
        if problems:
            result = f"Your model has {len(problems)} problems:\n"
            for p in problems:
                result += f"\n * {p}"
            if raise_exception:
                raise ModelError(result)
        else:
            result = "Your model is valid."
        return result

    def valid(self, parameters: ParameterGroup = None) -> bool:
        """Returns `True` if the number problems in the model is 0, else `False`

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        return len(self.problem_list(parameters)) == 0

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
        string += "_Megacomplex Types_: "
        string += ", ".join(self._megacomplex_types)
        string += "\n\n"

        for name in self.model_items:
            items = getattr(self, name)
            if not items:
                continue

            string += f"{base_heading}# {name.replace('_', ' ').title()}\n\n"

            if isinstance(items, dict):
                items = items.values()
            for item in items:
                item_str = item.mprint(
                    parameters=parameters, initial_parameters=initial_parameters
                ).split("\n")
                string += f"* {item_str[0]}\n"
                for s in item_str[1:]:
                    string += f"  {s}\n"
            string += "\n"
        return MarkdownStr(string)

    def _repr_markdown_(self) -> str:
        """Special method used by ``ipython`` to render markdown."""
        return str(self.markdown(base_heading_level=3))

    def __str__(self):
        return str(self.markdown())
