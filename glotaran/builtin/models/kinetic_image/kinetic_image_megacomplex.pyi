from typing import Any
from typing import List
from typing import Optional

from glotaran.model import model_attribute
from glotaran.parameter import Parameter

class KineticImageMegacomplex:
    @property
    def k_matrix(self) -> List[str]:
        ...

    @property
    def scale(self) -> Parameter:
        ...

    def full_k_matrix(self, model: Optional[Any] = ...):
        ...

    @property
    def involved_compartments(self):
        ...
