""" Glotaran Spectral Relation """
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import List
from typing import Tuple

import numpy as np
import xarray as xr

from glotaran.model import model_attribute
from glotaran.parameter import Parameter

if TYPE_CHECKING:
    from typing import Any

    from glotaran.parameter import ParameterGroup

    from .kinetic_spectrum_model import KineticSpectrumModel


@model_attribute(
    properties={
        "compartment": str,
        "target": str,
        "parameter": Parameter,
        "interval": List[Tuple[float, float]],
    },
    no_label=True,
)
class SpectralRelation:
    def applies(self, index: Any) -> bool:
        """
        Returns true if the index is in one of the intervals.

        Parameters
        ----------
        index :

        Returns
        -------
        applies : bool

        """
        return any(interval[0] <= index <= interval[1] for interval in self.interval)


def create_spectral_relation_matrix(
    model: KineticSpectrumModel,
    dataset: str,
    parameters: ParameterGroup,
    clp_labels: list[str],
    matrix: np.ndarray,
    index: float,
) -> tuple[list[str], np.ndarray]:
    relation_matrix = np.diagflat([1.0 for _ in clp_labels])

    idx_to_delete = []
    for relation in model.spectral_relations:
        if relation.target in clp_labels and relation.applies(index):

            if relation.compartment not in clp_labels:
                warnings.warn(
                    "Relation between compartments '{relation.compartment}' and "
                    f"'{relation.target}' cannot be applied for '{dataset}'. "
                    f" '{relation.source}' not in dataset"
                )
                continue

            relation = relation.fill(model, parameters)
            source_idx = clp_labels.index(relation.compartment)
            target_idx = clp_labels.index(relation.target)
            relation_matrix[target_idx, source_idx] = relation.parameter
            idx_to_delete.append(target_idx)

    clp_labels = [label for i, label in enumerate(clp_labels) if i not in idx_to_delete]
    relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
    return (clp_labels, relation_matrix)


def apply_spectral_relations(
    model: KineticSpectrumModel,
    dataset: str,
    parameters: ParameterGroup,
    clp_labels: list[str],
    matrix: np.ndarray,
    index: float,
) -> tuple[list[str], np.ndarray]:

    if not model.spectral_relations:
        return (clp_labels, matrix)

    reduced_clp_labels, relation_matrix = create_spectral_relation_matrix(
        model, dataset, parameters, clp_labels, matrix, index
    )

    reduced_matrix = matrix @ relation_matrix

    return (reduced_clp_labels, reduced_matrix)


def retrieve_related_clps(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: dict[str, list[str] | list[list[str]]],
    clps: dict[str, list[np.ndarray]],
    data: dict[str, xr.Dataset],
) -> dict[str, list[np.ndarray]]:

    for relation in model.spectral_relations:
        relation = relation.fill(model, parameters)
        for label, dataset_clp_labels in clp_labels.items():
            for i, index in enumerate(data[label].coords[model.global_dimension]):
                if (
                    relation.target in dataset_clp_labels[i]
                    and relation.compartment in dataset_clp_labels[i]
                    and relation.applies(index)
                ):
                    target_idx = dataset_clp_labels[i].index(relation.target)
                    source_idx = dataset_clp_labels[i].index(relation.compartment)
                    clps[label][i][target_idx] = clps[label][i][source_idx] * relation.parameter

    return clps
