"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import List

import numpy as np

from glotaran.builtin.megacomplexes.decay.decay_megacomplex_base import DecayMegacomplexBase
from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.model import DatasetModel
from glotaran.model import megacomplex
from glotaran.parameter import Parameter


@megacomplex(
    dimension="time",
    properties={
        "compartments": List[str],
        "rates": List[Parameter],
    },
    dataset_model_items={
        "irf": {"type": Irf, "allow_none": True},
    },
    register_as="decay-parallel",
)
class DecayParallelMegacomplex(DecayMegacomplexBase):
    def get_compartments(self, dataset_model: DatasetModel) -> list[str]:
        return self.compartments

    def get_initial_concentration(self, dataset_model: DatasetModel) -> np.ndarray:
        return np.ones((len(self.compartments)), dtype=np.float64)

    def get_k_matrix(self) -> KMatrix:
        size = len(self.compartments)
        k_matrix = KMatrix()
        k_matrix.matrix = {
            (self.compartments[i], self.compartments[i]): self.rates[i] for i in range(size)
        }
        return k_matrix
