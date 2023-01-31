import numpy as np
from numpy.typing import ArrayLike

from glotaran.model import EqualAreaPenalty
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix


def calculate_clp_penalties(
    matrices: list[OptimizationMatrix],
    estimations: list[OptimizationEstimation],
    global_axis: ArrayLike,
    penalties: list[EqualAreaPenalty],
) -> np.typing.ArrayLike:
    """Calculate the clp penalty.

    Parameters
    ----------
    clp_labels : list[list[str]]
        The clp labels.
    clps : list[np.typing.ArrayLike]
        The clps.
    global_axis : np.typing.ArrayLike
        The global axis.

    Returns
    -------
    list[float]
        The clp penalty.
    """
    sources = [[] for _ in penalties]
    targets = [[] for _ in penalties]
    for matrix, estimation, index in zip(matrices, estimations, global_axis):
        for i, penalty in enumerate(penalties):
            if penalty.source in matrix.clp_axis and penalty.source_applies(index):
                sources[i].append(estimation.clp[matrix.clp_axis.index(penalty.source)])
            if penalty.target in matrix.clp_axis and penalty.target_applies(index):
                targets[i].append(estimation.clp[matrix.clp_axis.index(penalty.target)])

    return np.array(
        [
            np.abs(np.sum(np.abs(source)) - penalty.parameter * np.sum(np.abs(target)))
            for penalty, source, target in zip(penalties, sources, targets)
        ]
    )
