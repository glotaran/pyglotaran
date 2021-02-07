from __future__ import annotations

from typing import TypeVar

from glotaran.model.base_model import Model
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

from .attribute import model_attribute  # noqa: F401

_T_Model = TypeVar("_T_Model", bound=Model)
_T_DatasetDescriptor = TypeVar("_T_DatasetDescriptor", bound=DatasetDescriptor)  # noqa: F821

class DatasetDescriptor:
    megacomplex: list[str]
    scale: Parameter | None = ...
    def fill(self, model: _T_Model, parameters: ParameterGroup) -> _T_DatasetDescriptor: ...
    @classmethod
    def from_dict(cls: type[_T_DatasetDescriptor], values: dict) -> _T_DatasetDescriptor: ...
    @classmethod
    def from_list(cls: type[_T_DatasetDescriptor], values: list) -> _T_DatasetDescriptor: ...
    def validate(self, model: _T_Model, parameters=...) -> list[str]: ...
    def mprint_item(
        self, parameters: ParameterGroup = ..., initial_parameters: ParameterGroup = ...
    ) -> str: ...
