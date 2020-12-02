from typing import Any
from typing import List
from typing import Type

from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed
from glotaran.parameter import Parameter

class IrfMeasured:
    ...


class IrfMultiGaussian:
    @property
    def center(self) -> List[Parameter]:
        ...

    @property
    def width(self) -> List[Parameter]:
        ...

    @property
    def scale(self) -> List[Parameter]:
        ...

    @property
    def backsweep_period(self) -> Parameter:
        ...

    def parameter(self, index: Any):
        ...

    def calculate(self, index: Any, axis: Any):
        ...


class IrfGaussian(IrfMultiGaussian):
    @property
    def center(self) -> Parameter:
        ...

    @property
    def width(self) -> Parameter:
        ...


class Irf:
    @classmethod
    def add_type(cls, type_name: str, type: Type) -> None:
        ...
