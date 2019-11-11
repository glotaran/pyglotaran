"""This package contains compartment constraint items."""

import typing
import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter, ParameterGroup

from .spectral_relations import retrieve_clps


@model_attribute(properties={
    'compartment': str,
    'interval': typing.List[typing.Tuple[float, float]],
    'target': str,
    'parameter': Parameter,
    'weight': str,
}, no_label=True)
class EqualAreaPenalty:
    """An equal area constraint adds a the differenc of the sum of a
    compartements in the e matrix in one ore more intervals to the scaled sum
    of the e matrix of one or more target compartmants to resiudal. The additional
    residual is scaled with the weight."""

    def applies(self, index: any) -> bool:
        """
        Returns true if the indexx is in one of the intervals.

        Parameters
        ----------
        index : any

        Returns
        -------
        applies : bool

        """
        def applies(interval):
            return interval[0] <= index <= interval[1]
        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any([applies(i) for i in self.interval])


def has_spectral_penalties(model: typing.Type['KineticModel']) -> bool:
    return len(model.equal_area_penalties) != 0


def apply_spectral_penalties(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        clps: np.ndarray,
        index: float) -> np.ndarray:

    clp_labels, clps = retrieve_clps(model, parameter, clp_labels, clps, index)

    penalties = []
    for penalty in model.equal_area_penalties:
        if penalty.applies(index):
            penalty = penalty.fill(model, parameter)
            source_idx = clp_labels.index(penalty.compartment)
            target_idx = clp_labels.index(penalty.target)
            penalties.append(
                (clps[source_idx] - penalty.parameter * clps[target_idx]) * penalty.weight
            )
    return penalties
