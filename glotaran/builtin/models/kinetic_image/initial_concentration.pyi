from typing import List

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute
from glotaran.parameter import Parameter

class InitialConcentration:
    @property
    def compartments(self) -> List[str]:
        ...

    @property
    def parameters(self) -> List[Parameter]:
        ...

    @property
    def exclude_from_normalize(self) -> List[Parameter]:
        ...

    def normalized(self, dataset: DatasetDescriptor) -> InitialConcentration:
        ...
