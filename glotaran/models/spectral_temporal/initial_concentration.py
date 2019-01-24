"""This package contains the intial concentration item."""
from dataclasses import replace

from typing import List
import numpy as np

from glotaran.model import model_item, DatasetDescriptor
from glotaran.parameter import Parameter


@model_item(
    attributes={'compartments': {'type': List[str], 'target': None}, 'parameters': List[Parameter]}
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
        parameters /= np.sum(parameters)
        return replace(self, parameters=parameters)
