"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay import DecayParallelMegacomplex
from glotaran.builtin.megacomplexes.decay.decay_parallel_megacomplex import DecayDatasetModel
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import calculate_matrix
from glotaran.builtin.megacomplexes.decay.util import finalize_data
from glotaran.model import DatasetModel
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@megacomplex(dataset_model_type=DecayDatasetModel)
class DecaySequentialMegacomplex(DecayParallelMegacomplex):
    """A Megacomplex with one or more K-Matrices."""

    type: str = "decay-sequential"

    def get_compartments(self, dataset_model: DatasetModel) -> list[str]:
        return self.compartments

    def get_initial_concentration(
        self, dataset_model: DatasetModel, normalized: bool = True
    ) -> np.ndarray:
        initial_concentration = np.zeros((len(self.compartments)), dtype=np.float64)
        initial_concentration[0] = 1
        return initial_concentration

    def get_k_matrix(self) -> KMatrix:
        k_matrix = KMatrix(
            label="",
            matrix={
                (self.compartments[i + 1], self.compartments[i]): self.rates[i]
                for i in range(len(self.compartments) - 1)
            },
        )
        k_matrix.matrix[self.compartments[-1], self.compartments[-1]] = self.rates[-1]
        return k_matrix

    def get_a_matrix(self, dataset_model: DatasetModel) -> np.ndarray:
        return self.get_k_matrix().a_matrix_sequential(self.get_compartments(dataset_model))

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        return calculate_matrix(self, dataset_model, global_axis, model_axis, **kwargs)

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        finalize_data(dataset_model, dataset, is_full_model, as_global)
