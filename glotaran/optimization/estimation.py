from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from glotaran.optimization.nnls import residual_nnls
from glotaran.optimization.variable_projection import residual_variable_projection

if TYPE_CHECKING:
    from glotaran.model.clp_relation import ClpRelation
    from glotaran.typing.types import ArrayLike

SUPPORTED_RESIDUAL_FUNCTIONS = {
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
        matrix : ArrayLike
            The matrix.
        data : ArrayLike
            The data.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The estimated clp and residual.
        """
        residual_fn = SUPPORTED_RESIDUAL_FUNCTIONS[residual_function]
        return cls(*residual_fn(matrix, data))

    def resolve_clp(
        self,
        clp_axis: list[str],
        reduced_clp_axis: list[str],
        index: float,
        relations: list[ClpRelation],
    ) -> OptimizationEstimation:
        clp = self.clp
        self.clp = np.zeros(len(clp_axis))
        self.clp[[clp_axis.index(label) for label in reduced_clp_axis]] = clp
        for relation in [r for r in relations if r.applies(index)]:
            source_idx = clp_axis.index(relation.source)
            target_idx = clp_axis.index(relation.target)
            self.clp[target_idx] = relation.parameter * self.clp[source_idx]
        return self
