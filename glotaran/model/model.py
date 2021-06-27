"""A base class for global analysis models."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from glotaran.deprecation import deprecate
from glotaran.model.attribute import model_attribute_typed
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.relation import Relation
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
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
        self._add_default_items()
        self._add_megacomplexe_types()

    def _add_megacomplexe_types(self):
        @model_attribute_typed({})
        class MetaMegacomplex:
            """This class holds all Megacomplex types defined by a model."""

        for name, megacomplex_type in self._megacomplex_types.items():
            if not issubclass(megacomplex_type, Megacomplex):
                raise TypeError(
                    f"Megacomplex type {name}(megacomplex_type) is not a subclass of Megacomplex"
                )
            MetaMegacomplex.add_type(name, megacomplex_type)
            self._add_megacomplex_type(megacomplex_type)

        setattr(
            MetaMegacomplex,
            "_glotaran_model_attribute_default_type",
            self.default_megacomplex,
        )
        self._add_model_item("megacomplex", MetaMegacomplex)

    def _add_megacomplex_type(self, megacomplex_type: type[Megacomplex]):

        for name, item in megacomplex_type.glotaran_model_items().items():

            if name in self._model_items:
                if self._model_items[name] != megacomplex_type:
                    raise ModelError(
                        f"Cannot add megacomplex of type. Model item '{name}' was already defined"
                        "by another megacomplex as a different type."
                    )
                continue
            self._add_model_item(name, item)

    def _add_model_item(self, name: str, item: object):
        self._model_items[name] = item
        print("Add item", name, item)

        if getattr(item, "_glotaran_has_label"):
            setattr(self, f"{name}", {})
        else:
            setattr(self, f"{name}", [])

    def _add_default_items(self):
        self._add_model_item("clp_area_penalties", EqualAreaPenalty)
        self._add_model_item("constraints", Constraint)
        self._add_model_item("relations", Relation)
        self._add_model_item("weights", Weight)

    @classmethod
    def from_dict(cls, model_dict_ref: dict) -> Model:
        """Creates a model from a dictionary.

        Parameters
        ----------
        model_dict :
            Dictionary containing the model.
        """

        model = cls()

        model_dict = copy.deepcopy(model_dict_ref)

        # iterate over items
        for name, attribute in list(model_dict.items()):

            # we determine if the item is known by the model by looking for
            # a setter with same name.

            if hasattr(model, f"set_{name}"):

                # get the set function
                model_set = getattr(model, f"set_{name}")

                for label, item in attribute.items():
                    # we retrieve the actual class from the signature
                    item_cls = model_set.__func__.__annotations__["item"]

                    is_typed = hasattr(item_cls, "_glotaran_model_attribute_typed")

                    if isinstance(item, dict):
                        if is_typed:
                            if "type" not in item and item_cls.get_default_type() is None:
                                raise ValueError(f"Missing type for attribute '{name}'")
                            item_type = item.get("type", item_cls.get_default_type())

                            types = item_cls._glotaran_model_attribute_types
                            if item_type not in types:
                                raise ValueError(
                                    f"Unknown type '{item_type}' for attribute '{name}'"
                                )
                            item_cls = types[item_type]
                        item["label"] = label
                        model_set(label, item_cls.from_dict(item))
                    elif isinstance(item, list):
                        if is_typed:
                            if len(item) < 2 and len(item) != 1:
                                raise ValueError(f"Missing type for attribute '{name}'")
                            item_type = item[0]
                            types = item_cls._glotaran_model_attribute_types

                            if item_type not in types:
                                raise ValueError(
                                    f"Unknown type '{item_type}' for attribute '{name}'"
                                )
                            item_cls = types[item_type]
                        item = [label] + item
                        model_set(label, item_cls.from_list(item))
                del model_dict[name]

            elif hasattr(model, f"add_{name}"):

                # get the set function
                add = getattr(model, f"add_{name}")

                # we retrieve the actual class from the signature
                for item in attribute:
                    item_cls = add.__func__.__annotations__["item"]
                    is_typed = hasattr(item_cls, "_glotaran_model_attribute_typed")
                    if isinstance(item, dict):
                        if is_typed:
                            if "type" not in item:
                                raise ValueError(f"Missing type for attribute '{name}'")
                            item_type = item["type"]

                            if item_type not in item_cls._glotaran_model_attribute_types:
                                raise ValueError(
                                    f"Unknown type '{item_type}' for attribute '{name}'"
                                )
                            item_cls = item_cls._glotaran_model_attribute_types[item_type]
                        add(item_cls.from_dict(item))
                    elif isinstance(item, list):
                        if is_typed:
                            if len(item) < 2 and len(item) != 1:
                                raise ValueError(f"Missing type for attribute '{name}'")
                            item_type = (
                                item[1]
                                if len(item) != 1 and hasattr(item_cls, "label")
                                else item[0]
                            )

                            if item_type not in item_cls._glotaran_model_attribute_types:
                                raise ValueError(
                                    f"Unknown type '{item_type}' for attribute '{name}'"
                                )
                            item_cls = item_cls._glotaran_model_attribute_types[item_type]
                        add(item_cls.from_list(item))
                del model_dict[name]

        return model

    @property
    def default_megacomplex(self) -> str:
        """The default megacomplex used by this model."""
        return self._default_megacomplex_type

    @property
    def megacomplex_types(self) -> dict[str, type[Megacomplex]]:
        """The megacomplex types used by this model."""
        return self._megacomplex_types

    def problem_list(self, parameters: ParameterGroup = None) -> list[str]:
        """
        Returns a list with all problems in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        problems = []

        attrs = getattr(self, "_glotaran_model_attributes")
        for attr in attrs:
            attr = getattr(self, attr)
            if isinstance(attr, list):
                for item in attr:
                    problems += item.validate(self, parameters=parameters)
            else:
                for _, item in attr.items():
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

    def validate(self, parameters: ParameterGroup = None) -> str:
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
        attrs = getattr(self, "_glotaran_model_attributes")
        string = f"{base_heading} Model\n\n"
        string += f"_Type_: {self.model_type}\n\n"

        for attr in attrs:
            child_attr = getattr(self, attr)
            if not child_attr:
                continue

            string += f"{base_heading}# {attr.replace('_', ' ').title()}\n\n"

            if isinstance(child_attr, dict):
                child_attr = child_attr.values()
            for item in child_attr:
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
