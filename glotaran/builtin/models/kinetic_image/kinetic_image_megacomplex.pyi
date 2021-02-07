from __future__ import annotations

from typing import Any

from glotaran.model import model_attribute  # noqa: F401
from glotaran.parameter import Parameter

class KineticImageMegacomplex:
    @property
    def k_matrix(self) -> list[str]: ...
    @property
    def scale(self) -> Parameter: ...
    def full_k_matrix(self, model: Any | None = ...): ...
    @property
    def involved_compartments(self): ...
