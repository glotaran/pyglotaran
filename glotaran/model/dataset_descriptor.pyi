from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar

from glotaran.model.base_model import Model
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

from .attribute import model_attribute

T_Model = TypeVar("T_Model", bound=Model)
T_DatasetDescriptor = TypeVar("T_DatasetDescriptor", bound=DatasetDescriptor)


class DatasetDescriptor:
    megacomplex: List[str]
    scale: Optional[Parameter] = None

    def fill(self, model: T_Model, parameters: ParameterGroup) -> T_DatasetDescriptor:
        ...

    @classmethod
    def from_dict(cls: Type[T_DatasetDescriptor], values: Dict) -> T_DatasetDescriptor:
        ...

    @classmethod
    def from_list(cls: Type[T_DatasetDescriptor], values: List) -> T_DatasetDescriptor:
        ...

    def validate(self, model: T_Model, parameters=None) -> List[str]:
        ...

    def mprint_item(self, parameters: ParameterGroup = None, initial_parameters: ParameterGroup = None) -> str:
        ...
