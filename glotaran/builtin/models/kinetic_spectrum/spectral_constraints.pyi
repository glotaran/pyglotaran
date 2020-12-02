from typing import Any
from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed

from .kinetic_spectrum_model import KineticSpectrumModel

class OnlyConstraint:
    @property
    def compartment(self) -> str:
        ...

    @property
    def interval(self) -> List[Tuple[float, float]]:
        ...

    def applies(self, index: Any) -> bool:
        ...


class ZeroConstraint:
    @property
    def compartment(self) -> str:
        ...

    @property
    def interval(self) -> List[Tuple[float, float]]:
        ...

    def applies(self, index: Any) -> bool:
        ...


class SpectralConstraint:
    ...


def apply_spectral_constraints(
    model: KineticSpectrumModel, clp_labels: List[str], matrix: np.ndarray, index: float
) -> Tuple[List[str], np.ndarray]:
    ...
