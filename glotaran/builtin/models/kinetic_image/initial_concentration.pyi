from __future__ import annotations

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute  # noqa: F401
from glotaran.parameter import Parameter

class InitialConcentration:
    @property
    def compartments(self) -> list[str]: ...
    @property
    def parameters(self) -> list[Parameter]: ...
    @property
    def exclude_from_normalize(self) -> list[Parameter]: ...
    def normalized(self, dataset: DatasetDescriptor) -> InitialConcentration: ...
