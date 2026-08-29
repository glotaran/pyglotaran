from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glotaran.model.clp_penalties import EqualAreaPenalty
    from glotaran.optimization.estimation import OptimizationEstimation
    from glotaran.optimization.matrix import OptimizationMatrix
    from glotaran.typing.types import ArrayLike


def _get_area(
    label: str,
    intervals: list[tuple[float, float]] | tuple[float, float] | None,
    matrices: list[OptimizationMatrix],
    estimations: list[OptimizationEstimation],
    global_axis: ArrayLike,
) -> ArrayLike:
    intervals = [(-np.inf, np.inf)] if intervals is None else intervals
    intervals = [intervals] if isinstance(intervals, tuple) else intervals
    area = []
    for lower, upper in intervals:
        if lower > global_axis[-1]:
            continue
        lower = max(lower, np.min(global_axis))
        upper = min(upper, np.max(global_axis))
        if lower > upper:
            lower, upper = upper, lower
        start = np.abs(global_axis - lower).argmin()
        stop = np.abs(global_axis - upper).argmin() + 1
        for matrix, estimation in zip(
            matrices[start:stop], estimations[start:stop], strict=True
        ):
            if label in matrix.clp_axis:
                area.append(estimation.clp[matrix.clp_axis.index(label)])
    return np.asarray(area)


def calculate_clp_penalties(
    matrices: list[OptimizationMatrix],
    estimations: list[OptimizationEstimation],
    global_axis: ArrayLike,
    penalties: list[EqualAreaPenalty],
) -> ArrayLike:
    """Calculate the clp penalty.

    Parameters
    ----------
    clp_labels : list[list[str]]
        The clp labels.
    clps : list[ArrayLike]
        The clps.
    global_axis : ArrayLike
        The global axis.

    Returns
    -------
    list[float]
        The clp penalty.
    """
    return np.array(
        [
            np.abs(
                np.sum(
                    _get_area(
                        penalty.source,
                        penalty.source_intervals,
                        matrices,
                        estimations,
                        global_axis,
                    )
                )
                - penalty.parameter
                * np.sum(
                    _get_area(
                        penalty.target,
                        penalty.target_intervals,
                        matrices,
                        estimations,
                        global_axis,
                    )
                )
            )
            * penalty.weight
            for penalty in penalties
        ]
    )
