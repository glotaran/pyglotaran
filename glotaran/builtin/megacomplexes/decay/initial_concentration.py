"""This package contains the initial concentration item."""
from __future__ import annotations

from typing import List

import numpy as np

from glotaran.model import model_item
from glotaran.parameter import Parameter


@model_item(
    properties={
        "compartments": List[str],
        "parameters": List[Parameter],
        "exclude_from_normalize": {"type": List[str], "default": []},
    }
)
class InitialConcentration:
    """An initial concentration describes the population of the compartments at
    the beginning of an experiment."""

    def normalized(self) -> np.ndarray:
        normalized = np.array(self.parameters)
        idx = [c not in self.exclude_from_normalize for c in self.compartments]
        normalized[idx] /= np.sum(normalized[idx])
        return normalized
