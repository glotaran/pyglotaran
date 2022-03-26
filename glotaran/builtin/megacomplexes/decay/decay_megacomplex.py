"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.initial_concentration import InitialConcentration
from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.util import calculate_matrix
from glotaran.builtin.megacomplexes.decay.util import finalize_data
from glotaran.builtin.megacomplexes.decay.util import index_dependent
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex


@megacomplex(
    dimension="time",
    model_items={
        "k_matrix": List[KMatrix],
    },
    properties={},
    dataset_model_items={
        "initial_concentration": {"type": InitialConcentration, "allow_none": True},
        "irf": {"type": Irf, "allow_none": True},
    },
    register_as="decay",
)
class DecayMegacomplex(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""

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
