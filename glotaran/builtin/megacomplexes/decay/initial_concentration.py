"""This package contains the initial concentration item."""
from __future__ import annotations

import numpy as np

from glotaran.model import ModelItem
from glotaran.model import ParameterType
from glotaran.model import item


@item
class InitialConcentration(ModelItem):
    """An initial concentration describes the population of the compartments at
    the beginning of an experiment."""

    compartments: list[str]
    parameters: list[ParameterType]
    exclude_from_normalize: list[str] = []

    def normalized(self) -> np.ndarray:
        normalized = np.array(self.parameters)
        idx = [c not in self.exclude_from_normalize for c in self.compartments]
        normalized[idx] /= np.sum(normalized[idx])
        return normalized
