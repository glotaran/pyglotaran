"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import calculate_matrix
from glotaran.builtin.megacomplexes.decay.util import finalize_data
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelItemType
from glotaran.model import ParameterType
from glotaran.model import item
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@item
class DecayDatasetModel(DatasetModel):
    irf: ModelItemType[Irf] | None = None


@megacomplex(dataset_model_type=DecayDatasetModel)
class DecayParallelMegacomplex(Megacomplex):
    dimension: str = "time"
    type: str = "decay-parallel"
    compartments: list[str]
    rates: list[ParameterType]

    def get_compartments(self, dataset_model: DatasetModel) -> list[str]:
        return self.compartments

    def get_initial_concentration(
        self, dataset_model: DatasetModel, normalized: bool = True
    ) -> np.ndarray:
        initial_concentration = np.ones((len(self.compartments)), dtype=np.float64)
        if normalized:
            initial_concentration /= initial_concentration.size
        return initial_concentration

    def get_k_matrix(self) -> KMatrix:
        return KMatrix(
            label="",
            matrix={
                (self.compartments[i], self.compartments[i]): self.rates[i]
                for i in range(len(self.compartments))
            },
        )

    def get_a_matrix(self, dataset_model: DatasetModel) -> np.ndarray:
        return self.get_k_matrix().a_matrix_general(
            self.get_compartments(dataset_model), self.get_initial_concentration(dataset_model)
        )

    def calculate_matrix(
        self,
        dataset_model: DecayDatasetModel,
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
