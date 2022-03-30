"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import calculate_matrix
from glotaran.builtin.megacomplexes.decay.util import finalize_data
from glotaran.builtin.megacomplexes.decay.util import index_dependent
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
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
    register_as="decay-sequential",
)
class DecaySequentialMegacomplex(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""

    def get_compartments(self, dataset_model: DatasetModel) -> list[str]:
        return self.compartments

    def get_initial_concentration(
        self, dataset_model: DatasetModel, normalized: bool = True
    ) -> np.ndarray:
        initial_concentration = np.zeros((len(self.compartments)), dtype=np.float64)
        initial_concentration[0] = 1
        return initial_concentration

    def get_k_matrix(self) -> KMatrix:
        size = len(self.compartments)
        k_matrix = KMatrix()
        k_matrix.matrix = {
            (self.compartments[i + 1], self.compartments[i]): self.rates[i]
            for i in range(size - 1)
        }
        k_matrix.matrix[self.compartments[-1], self.compartments[-1]] = self.rates[-1]
        return k_matrix

    def get_a_matrix(self, dataset_model: DatasetModel) -> np.ndarray:
        return self.get_k_matrix().a_matrix_sequential(self.get_compartments(dataset_model))

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return index_dependent(dataset_model)

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):
        return calculate_matrix(self, dataset_model, indices, **kwargs)

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        finalize_data(dataset_model, dataset, is_full_model, as_global)
