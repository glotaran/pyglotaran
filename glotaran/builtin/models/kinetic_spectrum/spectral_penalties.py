"""This package contains compartment constraint items."""

from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

from .spectral_relations import retrieve_clps

T_KineticSpectrumModel = TypeVar("glotaran.builtin.models.kinetic_spectrum.KineticSpectrumModel")


@model_attribute(
    properties={
        "compartment": str,
        "interval": List[Tuple[float, float]],
        "target": str,
        "parameter": Parameter,
        "weight": str,
    },
    no_label=True,
)
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


def has_spectral_penalties(model: T_KineticSpectrumModel) -> bool:
    return len(model.equal_area_penalties) != 0


def apply_spectral_penalties(
    model: T_KineticSpectrumModel,
    parameter: ParameterGroup,
    clp_labels: Union[List[str], List[List[str]]],
    clps: np.ndarray,
    global_axis: np.ndarray,
) -> np.ndarray:

    penalties = []
    for penalty in model.equal_area_penalties:

        source_area = []
        target_area = []

        penalty = penalty.fill(model, parameter)

        for interval in penalty.interval:

            start_idx, end_idx = _get_idx_from_interval(interval, global_axis)
            for i in range(start_idx, end_idx):

                index_clp_label = clp_labels
                index_clp = clps

                if model.index_dependent():
                    index_clp_label = index_clp_label[i]
                    index_clp = index_clp[i]

                index_clp_label, index_clp = retrieve_clps(
                    model, parameter, index_clp_label, index_clp, global_axis[i]
                )

                source_idx = index_clp_label.index(penalty.compartment)
                source_area.append(index_clp[source_idx])

                target_idx = index_clp_label.index(penalty.target)
                target_area.append(index_clp[target_idx])

        areaPenalty = np.sum(source_area) - penalty.parameter * np.sum(target_area)
        penalties.append(areaPenalty * penalty.weight)
    return penalties


def _get_idx_from_interval(interval, axis):
    start = np.abs(axis - interval[0]).argmin() if not np.isinf(interval[0]) else 0
    end = np.abs(axis - interval[1]).argmin() + 1 if not np.isinf(interval[1]) else axis.size
    return start, end
