from __future__ import annotations

from typing import Any

import numpy as np

from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_model import KineticSpectrumModel
from glotaran.model import model_attribute
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

class EqualAreaPenalty:
    @property
    def compartment(self) -> str: ...
    @property
    def interval(self) -> list[tuple[float, float]]: ...
    @property
    def target(self) -> str: ...
    @property
    def parameter(self) -> Parameter: ...
    @property
    def weight(self) -> str: ...
    def applies(self, index: Any) -> bool: ...

def has_spectral_penalties(model: KineticSpectrumModel) -> bool: ...
def apply_spectral_penalties(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: list[str] | list[list[str]],
    clps: np.ndarray,
    global_axis: np.ndarray,
) -> np.ndarray: ...
