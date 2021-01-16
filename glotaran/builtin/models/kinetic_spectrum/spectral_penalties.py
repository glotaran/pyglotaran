"""This package contains compartment constraint items."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import xarray as xr

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
    clp_labels: Dict[str, Union[List[str], List[List[str]]]],
    clps: Dict[str, List[np.ndarray]],
    matrices: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    data: Dict[str, xr.Dataset],
    group_tolerance: float,
) -> np.ndarray:

    penalties = []
    for penalty in model.equal_area_penalties:

        penalty = penalty.fill(model, parameter)
        source_area_indices, source_area = _get_area(
            model.index_dependent(),
            model.global_dimension,
            clp_labels,
            clps,
            data,
            group_tolerance,
            penalty,
            penalty.compartment,
        )

        target_area_indices, target_area = _get_area(
            model.index_dependent(),
            model.global_dimension,
            clp_labels,
            clps,
            data,
            group_tolerance,
            penalty,
            penalty.target,
        )

        if not np.isclose(source_area_indices[0], target_area_indices[0], atol=group_tolerance):
            if source_area_indices[0] < target_area_indices[0]:
                idx = np.argmin(np.abs(target_area_indices - source_area_indices[0]))
                source_area_indices = source_area_indices[idx:]
                source_area = source_area[idx:]
            else:
                idx = np.argmin(np.abs(source_area_indices - target_area_indices[0]))
                target_area_indices = target_area_indices[idx:]
                target_area = target_area[idx:]

        if not np.isclose(source_area_indices[-1], target_area_indices[-1], atol=group_tolerance):
            if source_area_indices[-1] < target_area_indices[-1]:
                idx = np.argmin(np.abs(target_area_indices - source_area_indices[-1]))
                target_area_indices = target_area_indices[:idx]
                target_area = target_area[:idx]
            else:
                idx = np.argmin(np.abs(source_area_indices - target_area_indices[-1]))
                source_area_indices = source_area_indices[:idx]
                source_area = source_area[:idx]

        if not source_area_indices.size == target_area_indices.size:
            if source_area_indices.size > target_area_indices.size:
                mask = [
                    np.any(np.isclose(target_area_indices, index)) for index in source_area_indices
                ]
                source_area_indices = source_area_indices[mask]
                source_area = source_area[mask]
            else:
                mask = [
                    np.any(np.isclose(source_area_indices, index, atol=group_tolerance))
                    for index in target_area_indices
                ]
                target_area_indices = target_area_indices[mask]
                target_area = target_area[mask]
        area_penalty = np.abs(np.sum(source_area) - penalty.parameter * np.sum(target_area))
        penalties.append(area_penalty * penalty.weight)
    return np.asarray(penalties)


def _get_area(
    index_dependent: bool,
    global_dimension: str,
    clp_labels: Dict[str, List[List[str]]],
    clps: Dict[str, List[np.ndarray]],
    data: Dict[str, xr.Dataset],
    group_tolerance: float,
    penalty: EqualAreaPenalty,
    compartment: str,
) -> Tuple[np.ndarray, np.ndarray]:
    area = []
    area_indices = []

    for label, dataset in data.items():
        global_axis = dataset.coords[global_dimension]
        for interval in penalty.interval:
            if interval[0] > global_axis[-1]:
                # interval not in this dataset
                continue

            start_idx, end_idx = _get_idx_from_interval(interval, global_axis)
            for i in range(start_idx, end_idx + 1):
                index_clp_labels = clp_labels[label][i] if index_dependent else clp_labels[label]
                if not len(area) == 0 and global_axis[i] - group_tolerance < area_indices[-1]:
                    # already got clp for this index
                    continue
                if compartment in index_clp_labels:
                    area.append(clps[label][i][index_clp_labels.index(compartment)])
                    area_indices.append(global_axis[i])

    return np.asarray(area_indices), np.asarray(area)


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
