"""This package contains the initial concentration item."""
from __future__ import annotations

import copy
import typing

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        "compartments": typing.List[str],
        "parameters": typing.List[Parameter],
        "exclude_from_normalize": {"type": typing.List[str], "default": []},
    }
)
class InitialConcentration:
    """An initial concentration describes the population of the compartments at
    the beginning of an experiment."""

    def normalized(self, dataset: DatasetDescriptor) -> InitialConcentration:
        parameters = self.parameters
        for megacomplex in dataset.megacomplex:
            scale = [
                megacomplex.scale
                if c in megacomplex.involved_compartments and megacomplex.scale
                else 1
                for c in self.compartments
            ]
            parameters = np.multiply(parameters, scale)
        idx = [c not in self.exclude_from_normalize for c in self.compartments]
        parameters[idx] /= np.sum(parameters[idx])
        new = copy.deepcopy(self)
        for i, value in enumerate(parameters):
            new.parameters[i].value = value
        return new
