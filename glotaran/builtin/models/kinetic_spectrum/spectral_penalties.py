"""This package contains compartment constraint items."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter

if TYPE_CHECKING:
    from typing import Any
    from typing import Union

    from glotaran.parameter import ParameterGroup

    from .kinetic_spectrum_model import KineticSpectrumModel


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

        def applies(interval):
            return interval[0] <= index <= interval[1]

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any([applies(i) for i in self.interval])


def has_spectral_penalties(model: KineticSpectrumModel) -> bool:
    return len(model.equal_area_penalties) != 0


def apply_spectral_penalties(
    model: KineticSpectrumModel,
    parameter: ParameterGroup,
    clp_labels: Union[List[str], List[List[str]]],
    full_clps: List[np.ndarray],
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

                # In case of an index dependent problem the clp_labels are per index
                index_clp_label = clp_labels[i] if model.index_dependent() else clp_labels

                index_clp = full_clps[i]

                source_idx = index_clp_label.index(penalty.compartment)
                source_area.append(index_clp[source_idx])

                target_idx = index_clp_label.index(penalty.target)
                target_area.append(index_clp[target_idx])

        area_penalty = np.abs(np.sum(source_area) - penalty.parameter * np.sum(target_area))
        penalties.append(area_penalty * penalty.weight)
    return penalties


def _get_idx_from_interval(interval, axis):
    start = np.abs(axis - interval[0]).argmin() if not np.isinf(interval[0]) else 0
    end = np.abs(axis - interval[1]).argmin() + 1 if not np.isinf(interval[1]) else axis.size
    return start, end
