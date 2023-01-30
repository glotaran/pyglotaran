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
) -> list[float]:
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
            if penalty.source in matrix.clp_labels and penalty.source_applies(index):
                sources[i].append(estimation.clp[matrix.clp_labels.index(penalty.source)])
            if penalty.target in matrix.clp_labels and penalty.target_applies(index):
                targets[i].append(estimation.clp[matrix.clp_labels.index(penalty.target)])

    return [
        np.sum(np.abs(np.array(source) - penalty.parameter * np.array(target)))
        for penalty, source, target in zip(penalties, sources, targets)
    ]
