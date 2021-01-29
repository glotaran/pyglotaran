from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter

from .initial_concentration import InitialConcentration

class KMatrix:
    @classmethod
    def empty(cls: Any, label: str, compartments: List[str]) -> KMatrix:
        ...

    def involved_compartments(self) -> List[str]:
        ...

    def combine(self, k_matrix: KMatrix) -> KMatrix:
        ...

    def matrix_as_markdown(self, compartments: List[str] = ..., fill_parameters: bool = ...) -> str:
        ...

    def a_matrix_as_markdown(self, initial_concentration: InitialConcentration) -> str:
        ...

    def reduced(self, compartments: List[str]) -> np.ndarray:
        ...

    def full(self, compartments: List[str]) -> np.ndarray:
        ...

    def eigen(self, compartments: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def rates(self, initial_concentration: InitialConcentration) -> np.ndarray:
        ...

    def a_matrix(self, initial_concentration: InitialConcentration) -> np.ndarray:
        ...

    def a_matrix_non_unibranch(self, initial_concentration: InitialConcentration) -> np.ndarray:
        ...

    def a_matrix_unibranch(self, initial_concentration: InitialConcentration) -> np.array:
        ...

    def is_unibranched(self, initial_concentration: InitialConcentration) -> bool:
        ...

    @property
    def matrix(self) -> Dict[Tuple[str, str], Parameter]:
        ...
