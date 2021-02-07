from __future__ import annotations

from typing import TypeVar

from glotaran.model.base_model import Model
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

from .attribute import model_attribute  # noqa: F401

T_Model = TypeVar("T_Model", bound=Model)
T_DatasetDescriptor = TypeVar("T_DatasetDescriptor", bound=DatasetDescriptor)  # noqa: F821

class DatasetDescriptor:
    megacomplex: list[str]
    scale: Parameter | None = None
    def fill(self, model: T_Model, parameters: ParameterGroup) -> T_DatasetDescriptor: ...
    @classmethod
    def from_dict(cls: type[T_DatasetDescriptor], values: dict) -> T_DatasetDescriptor: ...
    @classmethod
    def from_list(cls: type[T_DatasetDescriptor], values: list) -> T_DatasetDescriptor: ...
    def validate(self, model: T_Model, parameters=None) -> list[str]: ...
    def mprint_item(
        self, parameters: ParameterGroup = None, initial_parameters: ParameterGroup = None
    ) -> str: ...
