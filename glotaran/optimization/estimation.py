from __future__ import annotations

from dataclasses import dataclass

from glotaran.optimization.nnls import residual_nnls
from glotaran.optimization.variable_projection import residual_variable_projection
from glotaran.typing.types import ArrayLike

SUPPORTED_RESIUDAL_FUNCTIONS = {
    "variable_projection": residual_variable_projection,
    "non_negative_least_squares": residual_nnls,
}


@dataclass(frozen=True)
class OptimizationEstimation:
    clp: ArrayLike
    residual: ArrayLike

    @classmethod
    def calculate(cls, matrix: ArrayLike, data: ArrayLike) -> OptimizationEstimation:
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

        residual_function = SUPPORTED_RESIUDAL_FUNCTIONS[dataset_group.residual_function]
        return cls(*residual_function(matrix, data))
