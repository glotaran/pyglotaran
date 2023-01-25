from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.initial_concentration import InitialConcentration
from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import calculate_matrix
from glotaran.builtin.megacomplexes.decay.util import finalize_data
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import ModelItemType
from glotaran.model import item
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@item
class DecayDatasetModel(DatasetModel):
    initial_concentration: ModelItemType[InitialConcentration] | None = None
    irf: ModelItemType[Irf] | None = None


@megacomplex(dataset_model_type=DecayDatasetModel)
class DecayMegacomplex(Megacomplex):
    dimension: str = "time"
    type: str = "decay"
    k_matrix: list[ModelItemType[KMatrix]]

    def get_compartments(self, dataset_model: DatasetModel) -> list[str]:
        if dataset_model.initial_concentration is None:
            raise ModelError(
                f'No initial concentration specified in dataset "{dataset_model.label}"'
            )
        return [
            compartment
            for compartment in dataset_model.initial_concentration.compartments
            if compartment in self.get_k_matrix().involved_compartments()
        ]

    def get_initial_concentration(
        self, dataset_model: DatasetModel, normalized: bool = True
    ) -> np.ndarray:
        compartments = self.get_compartments(dataset_model)
        idx = [
            compartment in compartments
            for compartment in dataset_model.initial_concentration.compartments
        ]
        initial_concentration = (
            dataset_model.initial_concentration.normalized()
            if normalized
            else np.asarray(dataset_model.initial_concentration.parameters)
        )
        return initial_concentration[idx]

    def get_k_matrix(self) -> KMatrix:
        full_k_matrix = None
        for k_matrix in self.k_matrix:
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        return full_k_matrix

    def get_a_matrix(self, dataset_model: DatasetModel) -> np.ndarray:
        return self.get_k_matrix().a_matrix(
            self.get_compartments(dataset_model), self.get_initial_concentration(dataset_model)
        )

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
