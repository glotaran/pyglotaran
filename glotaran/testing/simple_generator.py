from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from glotaran.model import Model
from glotaran.parameter.parameter_group import ParameterGroup

if TYPE_CHECKING:
    from glotaran.utils.ipython import MarkdownStr


def _split_iterable_in_values_and_dicts(input_list: list) -> tuple[list, list]:
    values: list = [val for val in input_list if not isinstance(val, dict)]
    defaults: list = [val for val in input_list if isinstance(val, dict)]
    return values, defaults


class SimpleGeneratorError(Exception):
    """Exception raised for errors in the simple_generator.

    Attributes:
        generator -- generator which is invalid
        message -- explanation of the error
    """

    def __init__(self, message="Invalid generator state"):
        self.message = message
        super().__init__(self.message)


@dataclass
class SimpleGenerator:
    """A minimal boilerplate model and parameters generator"""

    rates: list[int | float] = field(default_factory=list)
    """A list of values representing decay rates"""
    k_matrix: Literal["parallel", "sequential"] | dict[tuple[str, str], str] = "parallel"
    """"A `dict` with a k_matrix specification or `Literal["parallel", "sequential"]`"""
    compartments: list[str] | None = None
    irf: dict[str, float] = field(default_factory=dict)
    initial_concentration: list[float] = field(default_factory=list)
    dispersion_coefficients: list[float] = field(default_factory=list)
    dispersion_center: float | None = None
    default_megacomplex: str = "decay"
    # TODO: add:
    # shapes: list[float] = field(default_factory=list, init=False)

    @property
    def valid(self) -> bool:
        try:
            return self.model.valid(parameters=self.parameters)
        except SimpleGeneratorError:
            return False

    def validate(self) -> str:
        return self.model.validate(parameters=self.parameters)

    @property
    def model(self) -> Model:
        return Model.from_dict(self.model_dict)

    @property
    def model_dict(self) -> dict:
        return self._model_dict()

    @property
    def parameters(self) -> ParameterGroup:
        return ParameterGroup.from_dict(self.parameters_dict)

    @property
    def parameters_dict(self) -> dict:
        return self._parameters_dict()

    @property
    def model_and_parameters(self) -> tuple[Model, ParameterGroup]:
        return self.model, self.parameters

    def _validate_rates(self) -> tuple[bool, str]:
        if not isinstance(self.rates, list):
            return False, f"generator.rates: must be a `list`, got: {self.rates}"
        if len(self.rates) == 0:
            return False, "generator.rates: must be a `list` with 1 or more rates"
        if not isinstance(self.rates[0], (int, float)):
            return False, f"generator.rates: 1st element must be numeric, got: {self.rates[0]}"
        return True, "generators.rates are valid"

    @property
    def _rates(self):
        rates_valid, message = self._validate_rates()
        if rates_valid:
            return _split_iterable_in_values_and_dicts(self.rates)
        else:
            raise SimpleGeneratorError(message=message)

    def _parameters_dict_items(self):
        rates, rates_defaults = self._rates
        items = {"rates": rates}
        if rates_defaults:
            items.update({"rates_defaults": rates_defaults[0]})
        items.update({"irf": [[key, value] for key, value in self.irf.items()]})
        if self.initial_concentration:
            items.update({"inputs": self.initial_concentration})
        elif self.k_matrix == "parallel":
            items.update(
                {
                    "inputs": [
                        ["1", 1],
                        {"vary": False},
                    ]
                }
            )
        elif self.k_matrix == "sequential":
            items.update(
                {
                    "inputs": [
                        ["1", 1],
                        ["0", 0],
                        {"vary": False},
                    ]
                }
            )
        return items

    def _model_dict_items(self) -> dict:
        rates, _ = self._rates
        nr = len(rates)
        indices = list(range(1, 1 + nr))
        items = {"default-megacomplex": self.default_megacomplex}
        if self.irf:
            items.update(
                {
                    "irf": {
                        "type": "multi-gaussian",
                        "center": ["irf.center"],
                        "width": ["irf.width"],
                    }
                }
            )
        if isinstance(self.k_matrix, dict):
            items.update({"k_matrix": self.k_matrix})
            items.update({"input_parameters": [f"inputs.{i}" for i in indices]})
            items.update({"compartments": [f"s{i}" for i in indices]})
            # TODO: get unique compartments from user defined k_matrix
        if self.k_matrix == "parallel":
            items.update({"input_parameters": ["inputs.1"] * nr})
            items.update({"k_matrix": {(f"s{i}", f"s{i}"): f"rates.{i}" for i in indices}})
        elif self.k_matrix == "sequential":
            items.update({"input_parameters": ["inputs.1"] + ["inputs.0"] * (nr - 1)})
            items.update(
                {"k_matrix": {(f"s{i if i==nr else i+1}", f"s{i}"): f"rates.{i}" for i in indices}}
            )
        if self.k_matrix in ("parallel", "sequential"):
            items.update({"compartments": [f"s{i}" for i in indices]})
        return items

    def _parameters_dict(self) -> dict:
        items = self._parameters_dict_items()
        rates = items["rates"]
        if "rates_defaults" in items:
            rates += [items["rates_defaults"]]
        result = {"rates": rates}
        if items["irf"]:
            result.update({"irf": items["irf"]})
        result.update({"inputs": items["inputs"]})
        return result

    def _model_dict(self) -> dict:
        items = self._model_dict_items()
        result = {"default-megacomplex": items["default-megacomplex"]}
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
            result.update(
                {
                    "irf": {
                        "irf1": items["irf"],
                    }
                }
            )
        return result

    def markdown(self) -> MarkdownStr:
        return self.model.markdown(parameters=self.parameters)
