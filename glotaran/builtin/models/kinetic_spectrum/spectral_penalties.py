"""This package contains compartment constraint items."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from glotaran.model import model_attribute
from glotaran.parameter import Parameter

if TYPE_CHECKING:
    from typing import Any
    from typing import Sequence
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
    compartments in the e matrix in one ore more intervals to the scaled sum
    of the e matrix of one or more target compartments to residual. The additional
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
    clp_labels: Union[
        Dict[str, List[str]],
        Dict[str, List[List[str]]],
        List[List[List[str]]],
    ],
    clps: Dict[str, List[np.ndarray]],
    global_axis: np.ndarray,
) -> np.ndarray:

    penalties = []
    for penalty in model.equal_area_penalties:

        source_area = []
        target_area = []

        penalty = penalty.fill(model, parameter)

        for label in clps:
            # get axis for label
            for interval in penalty.interval:

                start_idx, end_idx = _get_idx_from_interval(interval, global_axis)
                for i in range(start_idx, end_idx + 1):

                    # In case of an index dependent problem the clp_labels are per index
                    index_clp_label = clp_labels[i] if model.index_dependent() else clp_labels

                    index_clp = clps[label][i]

                    source_idx = index_clp_label[label].index(penalty.compartment)
                    source_area.append(index_clp[source_idx])

                    target_idx = index_clp_label[label].index(penalty.target)
                    target_area.append(index_clp[target_idx])

            area_penalty = np.abs(np.sum(source_area) - penalty.parameter * np.sum(target_area))
            penalties.append(area_penalty * penalty.weight)
    return penalties


def _get_idx_from_interval(
    interval: Tuple[float, float], axis: Union[Sequence[float], np.ndarray]
) -> Tuple[int, int]:
    """Retrieves start and end index of an interval on some axis

    Parameters
    ----------
    interval : A tuple of floats with begin and end of the interval
    axis : Array like object which can be cast to np.array

    Returns
    -------
    start, end : tuple of int

    """
    axis_array = np.array(axis)
    start = np.abs(axis_array - interval[0]).argmin() if not np.isinf(interval[0]) else 0
    end = (
        np.abs(axis_array - interval[1]).argmin()
        if not np.isinf(interval[1])
        else axis_array.size - 1
    )
    return start, end
