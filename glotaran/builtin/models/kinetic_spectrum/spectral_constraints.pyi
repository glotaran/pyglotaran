from __future__ import annotations

from typing import Any

import numpy as np

from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_model import KineticSpectrumModel
from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed

class OnlyConstraint:
    @property
    def compartment(self) -> str: ...
    @property
    def interval(self) -> list[tuple[float, float]]: ...
    def applies(self, index: Any) -> bool: ...

class ZeroConstraint:
    @property
    def compartment(self) -> str: ...
    @property
    def interval(self) -> list[tuple[float, float]]: ...
    def applies(self, index: Any) -> bool: ...

class SpectralConstraint: ...  # noqa: E701

def apply_spectral_constraints(
    model: KineticSpectrumModel, clp_labels: list[str], matrix: np.ndarray, index: float
) -> tuple[list[str], np.ndarray]: ...
