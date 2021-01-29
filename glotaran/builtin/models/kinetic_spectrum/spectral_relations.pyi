from typing import Any
from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

from .kinetic_spectrum_model import KineticSpectrumModel

class SpectralRelation:
    @property
    def compartment(self) -> str:
        ...

    @property
    def target(self) -> str:
        ...

    @property
    def parameter(self) -> Parameter:
        ...

    @property
    def interval(self) -> List[Tuple[float, float]]:
        ...

    def applies(self, index: Any) -> bool:
        ...


def create_spectral_relation_matrix(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: List[str],
    matrix: np.ndarray,
    index: float,
) -> Tuple[List[str], np.ndarray]:
    ...


def apply_spectral_relations(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: List[str],
    matrix: np.ndarray,
    index: float,
) -> Tuple[List[str], np.ndarray]:
    ...


def retrieve_related_clps(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: List[str],
    clps: np.ndarray,
    index: float,
) -> Tuple[List[str], np.ndarray]:
    ...
