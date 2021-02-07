from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Mapping
from typing import TypeVar

import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize  # noqa: F401
from glotaran.analysis.result import Result
from glotaran.analysis.scheme import Scheme  # noqa: F401
from glotaran.analysis.simulation import simulate  # noqa: F401
from glotaran.parameter import ParameterGroup

from .dataset_descriptor import DatasetDescriptor
from .decorator import FinalizeFunction
from .weight import Weight

_Cls = TypeVar("_Cls")

class Model:
    _model_type: str
    dataset: Mapping[str, DatasetDescriptor]
    megacomplex: Any
    weights: Weight
    model_dimension: str
    global_dimension: str
    global_matrix = None
    finalize_data: FinalizeFunction | None = ...
    grouped: Callable[[type[Model]], bool]
    index_dependent: Callable[[type[Model]], bool]
    @staticmethod
    def matrix(
        dataset_descriptor: DatasetDescriptor = ..., axis=..., index=...
    ) -> tuple[None, None] | tuple[list[Any], np.ndarray]: ...
    def add_megacomplex(self, item: Any): ...
    def add_weights(self, item: Weight): ...
    def get_dataset(self, label: str) -> DatasetDescriptor: ...
    @classmethod
    def from_dict(cls: type[_Cls], model_dict_ref: dict) -> _Cls: ...
    @property
    def index_dependent_matrix(self): ...
    @property
    def model_type(self) -> str: ...
    def simulate(  # noqa: F811
        self,
        dataset: str,
        parameters: ParameterGroup,
        axes: dict[str, np.ndarray] = ...,
        clp: np.ndarray | xr.DataArray = ...,
        noise: bool = ...,
        noise_std_dev: float = ...,
        noise_seed: int = ...,
    ) -> xr.Dataset: ...
    def result_from_parameter(
        self,
        parameters: ParameterGroup,
        data: dict[str, xr.DataArray | xr.Dataset],
        nnls: bool = ...,
        group_atol: float = ...,
    ) -> Result: ...
    def problem_list(self, parameters: ParameterGroup = ...) -> list[str]: ...
    def validate(self, parameters: ParameterGroup = ...) -> str: ...
    def valid(self, parameters: ParameterGroup = ...) -> bool: ...
    def markdown(self, parameters: ParameterGroup = ..., initial: ParameterGroup = ...) -> str: ...
