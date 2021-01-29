from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

from .kinetic_spectrum_model import KineticSpectrumModel

class EqualAreaPenalty:
    @property
    def compartment(self) -> str:
        ...

    @property
    def interval(self) -> List[Tuple[float, float]]:
        ...

    @property
    def target(self) -> str:
        ...

    @property
    def parameter(self) -> Parameter:
        ...

    @property
    def weight(self) -> str:
        ...

    def applies(self, index: Any) -> bool:
        ...


def has_spectral_penalties(model: KineticSpectrumModel) -> bool:
    ...


def apply_spectral_penalties(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: Union[List[str], List[List[str]]],
    clps: np.ndarray,
    global_axis: np.ndarray,
) -> np.ndarray:
    ...
