"""This package contains the decay megacomplex item."""
from __future__ import annotations

from typing import List

import numpy as np

from glotaran.builtin.megacomplexes.decay.decay_megacomplex_base import DecayMegacomplexBase
from glotaran.builtin.megacomplexes.decay.initial_concentration import InitialConcentration
from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.model import DatasetModel
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
class DecayMegacomplex(DecayMegacomplexBase):
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

    def get_initial_concentration(self, dataset_model: DatasetModel) -> np.ndarray:
        compartments = self.get_compartments(dataset_model)
        idx = [
            compartment in compartments
            for compartment in dataset_model.initial_concentration.compartments
        ]
        return dataset_model.initial_concentration.normalized()[idx]

    def get_k_matrix(self) -> KMatrix:
        full_k_matrix = None
        for k_matrix in self.k_matrix:
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        return full_k_matrix
