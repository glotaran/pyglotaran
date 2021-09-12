from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from rich import pretty
from rich import print

from glotaran.model import Model
from glotaran.parameter.parameter_group import ParameterGroup

pretty.install()


def _split_iterable_in_values_and_dicts(input) -> tuple[list, list]:
    values: list = [val for val in input if not isinstance(val, dict)]
    defaults: list = [val for val in input if isinstance(val, dict)]
    return values, defaults


@dataclass
class SimpleGenerator:
    rates: list[float] = field(default_factory=list)
    k_matrix: Literal["parallel", "sequential"] | dict[tuple[str, str], str] = "parallel"
    compartments: list[str] | None = None
    irf: dict[str, float] = field(default_factory=dict)
    initial_concentration: list[float] = field(default_factory=list)
    dispersion_coefficients: list[float] = field(default_factory=list)
    dispersion_center: float | None = None
    default_megacomplex: str = "decay"
    # shapes: list[float] = field(default_factory=list, init=False)

    # def __post_init__(self):
    #     self._parameters = {}

    @property
    def model(self) -> Model:
        return Model.from_dict(self.model_dict)

    @property
    def model_dict(self) -> dict:
        # return REF_MODEL_DICT
        return self._model_dict()

    @property
    def parameters(self) -> ParameterGroup:
        return ParameterGroup.from_dict(self.parameters_dict)

    @property
    def parameters_dict(self) -> dict:
        # return REF_PARAMETER_DICT
        return self._parameters_dict()

    @property
    def model_and_parameters(self):
        return self.model, self.parameters

    @property
    def _rates(self):
        return _split_iterable_in_values_and_dicts(self.rates)

    def _parameters_dict_items(self):
        items = {}
        rates, rates_defaults = self._rates
        items.update({"rates": rates})
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

    def _model_dict_items(self):
        rates, _ = self._rates
        nr = len(rates)
        indices = list(range(1, 1 + nr))
        items = {"default-megacomplex": self.default_megacomplex}
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

    def _parameters_dict(self):
        items = self._parameters_dict_items()
        return {
            "rates": [*items["rates"], items["rates_defaults"]],
            "irf": items["irf"],
            "inputs": items["inputs"],
        }

    def _model_dict(self):
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
        if items["irf"]:
            result["dataset"]["dataset1"].update({"irf": "irf1"})
            result.update(
                {
                    "irf": {
                        "irf1": items["irf"],
                    }
                }
            )
        return result

    def markdown(self):
        print(self.model.markdown(parameters=self.parameters))  # noqa T001


if __name__ == "__main__":
    generator = SimpleGenerator()
    generator.rates = [501e-3, 202e-4, 105e-5, {"non-negative": True, "vary": False}]
    generator.irf = {"center": 1.3, "width": 7.8}
    generator.k_matrix = "sequential"
    model, parameters = generator.model_and_parameters
    print(generator.markdown())  # noqa T001
