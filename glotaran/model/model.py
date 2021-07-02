"""A base class for global analysis models."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from typing import List
from warnings import warn

from glotaran.deprecation import deprecate
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
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr


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
        model_dict_ref: dict,
        *,
        megacomplex_types: dict[str, type[Megacomplex]],
        default_megacomplex_type: str | None = None,
    ) -> Model:
        """Creates a model from a dictionary.

        Parameters
        ----------
        model_dict :
            Dictionary containing the model.
        """

        model = cls(
            megacomplex_types=megacomplex_types, default_megacomplex_type=default_megacomplex_type
        )

        model_dict = copy.deepcopy(model_dict_ref)

        # iterate over items
        for name, items in list(model_dict.items()):

            if name not in model._model_items:
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
            item_cls = self._model_items[name]
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
            item_cls = self._model_items[name]
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
            if self._model_items[name] != item:
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
        self._add_model_item("clp_area_penalties", EqualAreaPenalty)
        self._add_model_item("constraints", Constraint)
        self._add_model_item("relations", Relation)
        self._add_model_item("weights", Weight)

        self._add_dataset_property("megacomplex", List[str])
        self._add_dataset_property(
            "megacomplex_scale", {"type": List[Parameter], "allow_none": True}
        )
        self._add_dataset_property("global_megacomplex", {"type": List[str], "default": []})
        self._add_dataset_property(
            "global_megacomplex_scale",
            {"type": List[Parameter], "default": None, "allow_none": True},
        )
        self._add_dataset_property(
            "scale", {"type": Parameter, "default": None, "allow_none": True}
        )

    def _add_dataset_type(self):
        dataset_model_type = create_dataset_model_type(self._dataset_properties)
        self._add_model_item("dataset", dataset_model_type)

    @property
    def default_megacomplex(self) -> str:
        """The default megacomplex used by this model."""
        return self._default_megacomplex_type

    @property
    def megacomplex_types(self) -> dict[str, type[Megacomplex]]:
        """The megacomplex types used by this model."""
        return self._megacomplex_types

    @property
    def global_megacomplex(self) -> dict[str, Megacomplex]:
        """Alias for `glotaran.model.megacomplex`. Needed internally."""
        return self.megacomplex

    def problem_list(self, parameters: ParameterGroup = None) -> list[str]:
        """
        Returns a list with all problems in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        problems = []

        for name in self._model_items:
            items = getattr(self, name)
            if isinstance(items, list):
                for item in items:
                    problems += item.validate(self, parameters=parameters)
            else:
                for _, item in items.items():
                    problems += item.validate(self, parameters=parameters)

        return problems

    @deprecate(
        deprecated_qual_name_usage="glotaran.model.base_model.Model.simulate",
        new_qual_name_usage="glotaran.analysis.simulation.simulate",
        to_be_removed_in_version="0.6.0",
        importable_indices=(2, 1),
    )
    def simulate(
        self,
        dataset: str,
        parameters: ParameterGroup,
        axes: dict[str, np.ndarray] = None,
        clp: np.ndarray | xr.DataArray = None,
        noise: bool = False,
        noise_std_dev: float = 1.0,
        noise_seed: int = None,
    ) -> xr.Dataset:
        """Simulates the model.

        Parameters
        ----------
        dataset :
            Label of the dataset to simulate.
        parameter :
            The parameters for the simulation.
        axes :
            A dictionary with axes for simulation.
        clp :
            Conditionally linear parameters. Used instead of `model.global_matrix` if provided.
        noise :
            If `True` noise is added to the simulated data.
        noise_std_dev :
            The standard deviation of the noise.
        noise_seed :
            Seed for the noise.
        """
        from glotaran.analysis.simulation import simulate

        return simulate(
            self,
            dataset,
            parameters,
            axes=axes,
            clp=clp,
            noise=noise,
            noise_std_dev=noise_std_dev,
            noise_seed=noise_seed,
        )

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
        string += ", ".join(name for name in self._megacomplex_types)
        string += "\n\n"

        for name in self._model_items:
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
