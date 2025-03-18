from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glotaran.model.clp_penalties import EqualAreaPenalty
    from glotaran.optimization.estimation import OptimizationEstimation
    from glotaran.optimization.matrix import OptimizationMatrix
    from glotaran.typing.types import ArrayLike


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
    sources: list[list[float]] = [[] for _ in penalties]
    targets: list[list[float]] = [[] for _ in penalties]
    for matrix, estimation, index in zip(matrices, estimations, global_axis, strict=False):
        for i, penalty in enumerate(penalties):
            if penalty.source in matrix.clp_axis and penalty.source_applies(index):
                sources[i].append(estimation.clp[matrix.clp_axis.index(penalty.source)])
            if penalty.target in matrix.clp_axis and penalty.target_applies(index):
                targets[i].append(estimation.clp[matrix.clp_axis.index(penalty.target)])

    return np.array(
        [
            np.abs(np.sum(np.abs(source)) - penalty.parameter * np.sum(np.abs(target)))
            * penalty.weight
            for penalty, source, target in zip(penalties, sources, targets, strict=True)
        ]
    )
