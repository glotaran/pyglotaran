from __future__ import annotations

from typing import Any

from glotaran.model import model_attribute  # noqa: F401
from glotaran.model import model_attribute_typed  # noqa: F401
from glotaran.parameter import Parameter

class IrfMeasured: ...  # noqa: E701

class IrfMultiGaussian:
    @property
    def center(self) -> list[Parameter]: ...
    @property
    def width(self) -> list[Parameter]: ...
    @property
    def scale(self) -> list[Parameter]: ...
    @property
    def backsweep_period(self) -> Parameter: ...
    def parameter(self, index: Any): ...
    def calculate(self, index: Any, axis: Any): ...

class IrfGaussian(IrfMultiGaussian):
    @property
    def center(self) -> Parameter: ...
    @property
    def width(self) -> Parameter: ...

class Irf:
    @classmethod
    def add_type(cls, type_name: str, attribute_type: type) -> None: ...
