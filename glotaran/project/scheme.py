from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import xarray as xr

from glotaran.model import Model
from glotaran.parameter import ParameterGroup


def _not_none(f):
    @functools.wraps(f)
    def decorator(self, value):
        if value is None:
            raise ValueError(f"{f.__name__} cannot be None")
        f(self, value)


@dataclass
class Scheme:
    model: Model
    parameters: ParameterGroup
    data: dict[str, xr.DataArray | xr.Dataset]
    group_tolerance: float = 0.0
    non_negative_least_squares: bool = False
    maximum_number_function_evaluations: int = None
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-8
    optimization_method: Literal[
        "TrustRegionReflection",
        "Dogbox",
        "Levenberg-Marquardt",
    ] = "TrustRegionReflection"

    def problem_list(self) -> list[str]:
        """Returns a list with all problems in the model and missing parameters."""
        return self.model.problem_list(self.parameters)

    def validate(self) -> str:
        """Returns a string listing all problems in the model and missing parameters."""
        return self.model.validate(self.parameters)

    def valid(self, parameters: ParameterGroup = None) -> bool:
        """Returns `True` if there are no problems with the model or the parameters,
        else `False`."""
        return self.model.valid(parameters)

    def markdown(self):
        s = self.model.markdown(parameters=self.parameters)

        s += "\n\n"
        s += "__Scheme__\n\n"

        s += f"* *nnls*: {self.nnls}\n"
        s += f"* *nfev*: {self.nfev}\n"
        s += f"* *group_tolerance*: {self.group_tolerance}\n"

        return s
