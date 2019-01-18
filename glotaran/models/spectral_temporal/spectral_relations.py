""" Glotaran Spectral Relation """

import typing
import numpy as np

from glotaran.model import model_item


@model_item(
    attributes={
        'compartment': str,
        'target': str,
        'parameter': str,
        'interval': typing.List[typing.Tuple[any, any]],
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


def apply_spectral_relations(model: typing.Type['glotaran.models.spectral_temporal.KineticModel'],
                             clp_labels: typing.List[str],
                             matrix: np.ndarray,
                             index: float):
    for relation in model.spectral_relations:
        if relation.applies(index):

            source_idx = clp_labels.index(relation.compartment)
            target_idx = clp_labels.index(relation.target)
            matrix[:, target_idx] += relation.parameter * matrix[:, source_idx]

            idx = [not label == relation.compartment for label in clp_labels]
            clp_labels = [label for label in clp_labels if not label == relation.compartment]
            matrix = matrix[:, idx]
    return (clp_labels, matrix)
