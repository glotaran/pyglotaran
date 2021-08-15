from __future__ import annotations

from typing import Any
from typing import Union

import xarray as xr

from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.dataset_model import DatasetModel
from glotaran.model.dataset_model import create_dataset_model_type
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import create_model_megacomplex_type
from glotaran.model.relation import Relation
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup
from glotaran.utils.ipython import MarkdownStr

default_model_items: Any
default_dataset_properties: Any

class Model:
    def __init__(
        self,
        megacomplex_types: dict[str, type[Megacomplex]],
        *,
        default_megacomplex_type: str | None = ...,
    ) -> None: ...
    @classmethod
    def from_dict(
        cls,
        model_dict_ref: dict,
        megacomplex_types: dict[str, type[Megacomplex]],
        *,
        default_megacomplex_type: str | None = ...,
    ) -> Model: ...
    def as_dict(self) -> dict: ...
    @property
    def default_megacomplex(self) -> str: ...
    @property
    def megacomplex_types(self) -> dict[str, type[Megacomplex]]: ...
    @property
    def model_items(self) -> dict[str, type[object]]: ...
    @property
    def global_megacomplex(self) -> dict[str, Megacomplex]: ...
    def need_index_dependent(self) -> bool: ...
    def is_groupable(self, parameters: ParameterGroup, data: dict[str, xr.DataArray]) -> bool: ...
    def problem_list(self, parameters: ParameterGroup = ...) -> list[str]: ...
    def validate(self, parameters: ParameterGroup = ..., raise_exception: bool = ...) -> str: ...
    def valid(self, parameters: ParameterGroup = ...) -> bool: ...
    def get_parameters(self) -> list[str]: ...
    def markdown(
        self,
        parameters: ParameterGroup = ...,
        initial_parameters: ParameterGroup = ...,
        base_heading_level: int = ...,
    ) -> MarkdownStr: ...
    @property
    def dataset(self) -> dict[str, DatasetModel]: ...
