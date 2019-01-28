"""This package contains the intial concentration item."""
from dataclasses import replace

from typing import List
import numpy as np

from glotaran.model import model_item, DatasetDescriptor
from glotaran.parameter import Parameter


@model_item(
    attributes={
        'compartments': List[str],
        'parameters': List[Parameter],
        'exclude_from_normalize': {'type': List[str], 'default': []},
    }
)
class InitialConcentration:
    """An initial concentration describes the population of the compartments at
    the beginning of an experiment."""

    def normalized(self, dataset: DatasetDescriptor) -> 'InitialConcentration':
        parameters = self.parameters
        for megacomplex in dataset.megacomplex:
            scale = [megacomplex.scale if c in megacomplex.involved_compartments and
                     megacomplex.scale else 1 for c in self.compartments]
            parameters = np.multiply(parameters, scale)
        idx = [c not in self.exclude_from_normalize for c in self.compartments]
        parameters[idx] /= np.sum(parameters[idx])
        return replace(self, parameters=parameters)
