from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from glotaran.model import ClpRelation
from glotaran.optimization.nnls import residual_nnls
from glotaran.optimization.variable_projection import residual_variable_projection
from glotaran.typing.types import ArrayLike

SUPPORTED_RESIUDAL_FUNCTIONS = {
    "variable_projection": residual_variable_projection,
    "non_negative_least_squares": residual_nnls,
}


@dataclass
class OptimizationEstimation:
    clp: ArrayLike
    residual: ArrayLike

    @classmethod
    def calculate(
        cls,
        matrix: ArrayLike,
        data: ArrayLike,
        residual_function: Literal["variable_projection", "non_negative_least_squares"],
    ) -> OptimizationEstimation:
        """Calculate the clps and the residual for a matrix and data.

        Parameters
        ----------
        matrix : np.typing.ArrayLike
            The matrix.
        data : np.typing.ArrayLike
            The data.

        Returns
        -------
        tuple[np.typing.ArrayLike, np.typing.ArrayLike]
            The estimated clp and residual.
        """
        residual_fn = SUPPORTED_RESIUDAL_FUNCTIONS[residual_function]
        return cls(*residual_fn(matrix, data))

    def resolve_clp(
        self,
        clp_axis: list[str],
        reduced_clp_axis: list[str],
        index: float,
        relations: list[ClpRelation],
    ) -> OptimizationEstimation:
        if len(relations) == 0:
            return self

        clps = np.zeros(len(clp_axis))
        clps[[clp_axis.index(label) for label in reduced_clp_axis]]

        for relation in relations:
            if (
                relation.target in clp_axis
                and relation.source in clp_axis
                and relation.applies(index)
            ):
                source_idx = clp_axis.index(relation.source)
                target_idx = clp_axis.index(relation.target)
                clps[target_idx] = relation.parameter * clps[source_idx]
        return clps
