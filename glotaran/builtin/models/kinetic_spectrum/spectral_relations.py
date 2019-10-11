""" Glotaran Spectral Relation """

import typing
import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter, ParameterGroup


@model_attribute(
    properties={
        'compartment': str,
        'target': str,
        'parameter': Parameter,
        'interval': typing.List[typing.Tuple[float, float]],
    }, no_label=True)
class SpectralRelation:
    def applies(self, index: any) -> bool:
        """
        Returns true if the index is in one of the intervals.

        Parameters
        ----------
        index : any

        Returns
        -------
        applies : bool

        """
        return any(interval[0] <= index <= interval[1] for interval in self.interval)


def create_spectral_relation_matrix(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float) -> typing.Tuple[typing.List[str], np.ndarray]:
    relation_matrix = np.diagflat([1.0 for _ in clp_labels])

    idx_to_delete = []
    for relation in model.spectral_relations:
        if relation.compartment in clp_labels and relation.applies(index):
            relation = relation.fill(model, parameter)
            source_idx = clp_labels.index(relation.compartment)
            target_idx = clp_labels.index(relation.target)
            relation_matrix[target_idx, source_idx] = relation.parameter
            idx_to_delete.append(target_idx)

    clp_labels = [label for i, label in enumerate(clp_labels) if i not in idx_to_delete]
    relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
    return (clp_labels, relation_matrix)


def apply_spectral_relations(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float) -> typing.Tuple[typing.List[str], np.ndarray]:

    if not model.spectral_relations:
        return (clp_labels, matrix)

    reduced_clp_labels, relation_matrix = \
        create_spectral_relation_matrix(model, parameter, clp_labels, matrix, 1)

    reduced_matrix = matrix @ relation_matrix

    return (reduced_clp_labels, reduced_matrix)


def retrieve_clps(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        reduced_clp_labels: typing.List[str],
        reduced_clps: np.ndarray,
        index: float) -> typing.Tuple[typing.List[str], np.ndarray]:

    if not model.spectral_relations:
        return reduced_clp_labels, reduced_clps

    retrieved_clp_labels = []
    retrieved_clps = []

    for relation in model.spectral_relations:
        if relation.compartment in reduced_clp_labels and relation.applies(index):
            relation = relation.fill(model, parameter)
            retrieved_clp_labels.append(relation.target)
            source_idx = reduced_clp_labels.index(relation.compartment)
            retrieved_clps.append(
                reduced_clps[source_idx] * relation.parameter
            )

    retrieved_clps = \
        np.concatenate([reduced_clps, retrieved_clps]) if retrieved_clps else reduced_clps

    return reduced_clp_labels + retrieved_clp_labels, retrieved_clps
