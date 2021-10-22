"""Model generators used to generate simple models from a set of inputs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from glotaran.model import Model
from glotaran.parameter.parameter_group import ParameterGroup

if TYPE_CHECKING:
    from glotaran.utils.ipython import MarkdownStr


def _split_iterable_in_non_dict_and_dict_items(
    input_list: list[float, dict[str, bool | float]],
) -> tuple[list[float], list[dict[str, bool | float]]]:
    """Split an iterable (list) into non-dict and dict items.

    Parameters
    ----------
    input_list : list[float, dict[str, bool | float]]
        A list of values of type `float` and a dict with parameter options, e.g.
        `[1, 2, 3, {"vary": False, "non-negative": True}]`

    Returns
    -------
    tuple[list[float], list[dict[str, bool | float]]]
        Split a list into non-dict (`values`) and dict items (`defaults`),
        return a tuple (`values`, `defaults`)
    """
    values: list = [val for val in input_list if not isinstance(val, dict)]
    defaults: list = [val for val in input_list if isinstance(val, dict)]
    return values, defaults


@dataclass
class SimpleModelGenerator:
    """A minimal boilerplate model and parameters generator.

    Generates a model (together with the parameters specification) based on
    parameter input values assigned to the generator's attributes
    """

    rates: list[float] = field(default_factory=list)
    """A list of values representing decay rates"""
    k_matrix: Literal["parallel", "sequential"] | dict[tuple[str, str], str] = "parallel"
    """"A `dict` with a k_matrix specification or `Literal["parallel", "sequential"]`"""
    compartments: list[str] | None = None
    """A list of compartment names"""
    irf: dict[str, float] = field(default_factory=dict)
    """A dict of items specifying an irf"""
    initial_concentration: list[float] = field(default_factory=list)
    """A list values representing the initial concentration"""
    dispersion_coefficients: list[float] = field(default_factory=list)
    """A list of values representing the dispersion coefficients"""
    dispersion_center: float | None = None
    """A value representing the dispersion center"""
    default_megacomplex: str = "decay"
    """The default_megacomplex identifier"""
    # TODO: add support for a spectral model:
    # shapes: list[float] = field(default_factory=list, init=False)

    @property
    def valid(self) -> bool:
        """Check if the generator state is valid.

        Returns
        -------
        bool
            Generator state obtained by calling the generated model's
            `valid` function with the generated parameters as input.
        """
        try:
            return self.model.valid(parameters=self.parameters)
        except ValueError:
            return False

    def validate(self) -> str:
        """Call `validate` on the generated model and return its output.

        Returns
        -------
        str
            A string listing problems in the generated model and parameters if any.
        """
        return self.model.validate(parameters=self.parameters)

    @property
    def model(self) -> Model:
        """Return the generated model.

        Returns
        -------
        Model
            The generated model of type :class:`glotaran.model.Model`.
        """
        return Model.from_dict(self.model_dict)

    @property
    def model_dict(self) -> dict:
        """Return a dict representation of the generated model.

        Returns
        -------
        dict
            A dict representation of the generated model.
        """
        return self._model_dict()

    @property
    def parameters(self) -> ParameterGroup:
        """Return the generated parameters of type :class:`glotaran.parameter.ParameterGroup`.

        Returns
        -------
        ParameterGroup
            The generated parameters of type of type :class:`glotaran.parameter.ParameterGroup`.
        """
        return ParameterGroup.from_dict(self.parameters_dict)

    @property
    def parameters_dict(self) -> dict:
        """Return a dict representation of the generated parameters.

        Returns
        -------
        dict
            A dict representing the generated parameters.
        """
        return self._parameters_dict()

    @property
    def model_and_parameters(self) -> tuple[Model, ParameterGroup]:
        """Return generated model and parameters.

        Returns
        -------
        tuple[Model, ParameterGroup]
            A model of type :class:`glotaran.model.Model` and
            and parameters of type :class:`glotaran.parameter.ParameterGroup`.
        """
        return self.model, self.parameters

    @property
    def _rates(self) -> tuple[list[float], list[dict[str, bool | float]]]:
        """Validate input to rates, return a tuple of rates and parameter defaults.

        Returns
        -------
        tuple[list[float], list[dict[str, bool | float]]]
            A tuple of a list of rates and a dict containing parameter defaults

        Raises
        ------
        ValueError
            Raised if rates is not a list of at least one number.
        """
        if not isinstance(self.rates, list):
            raise ValueError(f"generator.rates: must be a `list`, got: {self.rates}")
        if len(self.rates) == 0:
            raise ValueError("generator.rates: must be a `list` with 1 or more rates")
        if not isinstance(self.rates[0], (int, float)):
            raise ValueError(f"generator.rates: 1st element must be numeric, got: {self.rates[0]}")
        return _split_iterable_in_non_dict_and_dict_items(self.rates)

    def _parameters_dict_items(self) -> dict:
        """Return a dict with items used in constructing the parameters.

        Returns
        -------
        dict
            A dict with items used in constructing a parameters dict.
        """
        rates, rates_defaults = self._rates
        items = {"rates": rates}
        if rates_defaults:
            items["rates_defaults"] = rates_defaults[0]
        items["irf"] = [[key, value] for key, value in self.irf.items()]
        if self.initial_concentration:
            items["inputs"] = self.initial_concentration
        elif self.k_matrix == "parallel":
            items["inputs"] = [
                ["1", 1],
                {"vary": False},
            ]
        elif self.k_matrix == "sequential":
            items["inputs"] = [
                ["1", 1],
                ["0", 0],
                {"vary": False},
            ]
        return items

    def _model_dict_items(self) -> dict:
        """Return a dict with items used in constructing the model.

        Returns
        -------
        dict
            A dict with items used in constructing a model dict.
        """
        rates, _ = self._rates
        nr = len(rates)
        indices = list(range(1, 1 + nr))
        items = {"default_megacomplex": self.default_megacomplex}
        if self.irf:
            items["irf"] = {
                "type": "multi-gaussian",
                "center": ["irf.center"],
                "width": ["irf.width"],
            }
        if isinstance(self.k_matrix, dict):
            items["k_matrix"] = self.k_matrix
            items["input_parameters"] = [f"inputs.{i}" for i in indices]
            items["compartments"] = [f"s{i}" for i in indices]
            # TODO: get unique compartments from user defined k_matrix
        if self.k_matrix == "parallel":
            items["input_parameters"] = ["inputs.1"] * nr
            items["k_matrix"] = {(f"s{i}", f"s{i}"): f"rates.{i}" for i in indices}
        elif self.k_matrix == "sequential":
            items["input_parameters"] = ["inputs.1"] + ["inputs.0"] * (nr - 1)
            items["k_matrix"] = {
                (f"s{i if i==nr else i+1}", f"s{i}"): f"rates.{i}" for i in indices
            }

        if self.k_matrix in ("parallel", "sequential"):
            items["compartments"] = [f"s{i}" for i in indices]
        return items

    def _parameters_dict(self) -> dict:
        """Return a parameters dict.

        Returns
        -------
        dict
            A dict that can be passed to the `ParameterGroup` `from_dict` method.
        """
        items = self._parameters_dict_items()
        rates = items["rates"]
        if "rates_defaults" in items:
            rates += [items["rates_defaults"]]
        result = {"rates": rates}
        if items["irf"]:
            result["irf"] = items["irf"]
        result["inputs"] = items["inputs"]
        return result

    def _model_dict(self) -> dict:
        """Return a model dict.

        Returns
        -------
        dict
            A dict that can be passed to the `Model` `from_dict` method.
        """
        items = self._model_dict_items()
        result = {"default_megacomplex": items["default_megacomplex"]}
        result.update(
            {
                "initial_concentration": {
                    "j1": {
                        "compartments": items["compartments"],
                        "parameters": items["input_parameters"],
                    },
                },
                "megacomplex": {
                    "mc1": {"k_matrix": ["k1"]},
                },
                "k_matrix": {"k1": {"matrix": items["k_matrix"]}},
                "dataset": {
                    "dataset1": {
                        "initial_concentration": "j1",
                        "megacomplex": ["mc1"],
                    },
                },
            }
        )
        if "irf" in items:
            result["dataset"]["dataset1"].update({"irf": "irf1"})
            result["irf"] = {
                "irf1": items["irf"],
            }
        return result

    def markdown(self) -> MarkdownStr:
        """Return a markdown string representation of the generated model and parameters.

        Returns
        -------
        MarkdownStr
            A markdown string
        """
        return self.model.markdown(parameters=self.parameters)
