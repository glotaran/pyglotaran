"""The result class for global analysis."""

from __future__ import annotations

# TODO: Fix circular import
#  from glotaran import __version__ as glotaran_version
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict

from glotaran.optimization.optimization_history import OptimizationHistory  # noqa: TCH001
from glotaran.parameter import ParameterHistory  # noqa: TCH001

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

    from glotaran.parameter import Parameters
    from glotaran.typing.types import ArrayLike


class OptimizationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    """The result of a global analysis."""

    number_of_function_evaluations: int
    """The number of function evaluations."""

    success: bool
    """Indicates if the optimization was successful."""

    termination_reason: str
    """The reason (message when) the optimizer terminated"""

    #  glotaran_version: str
    #  """The glotaran version used to create the result."""

    free_parameter_labels: list[str]
    """List of labels of the free parameters used in optimization."""

    parameter_history: ParameterHistory
    """The parameter history."""

    optimization_history: OptimizationHistory
    """The optimization history."""

    # The below can be none in case of unsuccessful optimization

    cost: float | None = None
    """The final cost."""

    chi_square: float | None = None
    r"""The chi-square of the optimization.

    :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`."""

    covariance_matrix: np.ndarray | None = None
    """Covariance matrix.

    The rows and columns are corresponding to :attr:`free_parameter_labels`."""

    number_of_free_clp: int | None = None
    degrees_of_freedom: int | None = None
    """Degrees of freedom in optimization :math:`N - N_{vars}`."""

    jacobian: np.ndarray | list | None = None
    """Modified Jacobian matrix at the solution

    See also: :func:`scipy.optimize.least_squares`
    """
    number_of_data_points: int | None = None
    """Number of data points :math:`N`."""
    number_of_jacobian_evaluations: int | None = None
    """The number of jacobian evaluations."""
    number_of_parameters: int | None = None
    """Number of parameters in optimization :math:`N_{vars}`"""
    optimality: float | None = None
    reduced_chi_square: float | None = None
    r"""The reduced chi-square of the optimization.

    :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars})}`.
    """
    root_mean_square_error: float | None = None
    r"""
    The root mean square error the optimization.

    :math:`rms = \sqrt{\chi^2_{red}}`
    """

    @classmethod
    def from_least_squares_result(
        cls,
        result: OptimizeResult | None,
        parameter_history: ParameterHistory,
        optimization_history: OptimizationHistory,
        penalty: ArrayLike,
        free_parameter_labels: list[str],
        termination_reason: str,
        number_of_free_clp: int,
    ):
        success = result is not None

        result_args = {
            "success": success,
            "free_parameter_labels": free_parameter_labels,
            "parameter_history": parameter_history,
            "termination_reason": termination_reason,
            "optimization_history": optimization_history,
            "number_of_function_evaluations": result.nfev  # type:ignore[union-attr]
            if success
            else parameter_history.number_of_records,
            "cost": 0.5 * np.dot(penalty, penalty),
        }

        if success:
            result_args["number_of_free_clp"] = number_of_free_clp
            result_args["number_of_jacobian_evaluations"] = result.njev  # type:ignore[union-attr]
            result_args["optimality"] = float(result.optimality)  # type:ignore[union-attr]
            result_args["number_of_data_points"] = result.fun.size  # type:ignore[union-attr]
            result_args["number_of_parameters"] = result.x.size  # type:ignore[union-attr]
            result_args["degrees_of_freedom"] = (
                result_args["number_of_data_points"]
                - result_args["number_of_parameters"]
                - result_args["number_of_free_clp"]
            )
            result_args["chi_square"] = float(np.sum(result.fun**2))  # type:ignore[union-attr]
            result_args["reduced_chi_square"] = (
                result_args["chi_square"] / result_args["degrees_of_freedom"]
            )
            result_args["root_mean_square_error"] = float(
                np.sqrt(result_args["reduced_chi_square"])
            )
            result_args["jacobian"] = result.jac  # type:ignore[union-attr]
            result_args["covariance_matrix"] = calculate_covariance_matrix_and_standard_errors(
                result_args["jacobian"], result_args["root_mean_square_error"]
            )

        return cls(**result_args)

    def calculate_parameter_errors(self, parameters: Parameters):
        if self.covariance_matrix is not None:
            standard_errors = self.root_mean_square_error * np.sqrt(
                np.diag(self.covariance_matrix)
            )
            for label, error in zip(self.free_parameter_labels, standard_errors):
                parameters.get(label).standard_error = error


def calculate_covariance_matrix_and_standard_errors(
    jacobian: ArrayLike, root_mean_square_error: float
) -> ArrayLike:
    """Calculate the covariance matrix and standard errors of the optimization.

    Parameters
    ----------
    jacobian : ArrayLike
        The jacobian matrix.
    root_mean_square_error : float
        The root mean square error.

    Returns
    -------
    ArrayLike
        The covariance matrix.
    """
    # See PR #706: More robust covariance matrix calculation
    _, jacobian_sv, jacobian_rsv = np.linalg.svd(jacobian, full_matrices=False)
    jacobian_sv_square = jacobian_sv**2
    mask = jacobian_sv_square > np.finfo(float).eps
    return (jacobian_rsv[mask].T / jacobian_sv_square[mask]) @ jacobian_rsv[mask]
