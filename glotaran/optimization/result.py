"""The result class for global analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult

from glotaran import __version__ as glotaran_version
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
from glotaran.parameter import Parameters
from glotaran.project.dataclass_helpers import exclude_from_dict_field
from glotaran.project.dataclass_helpers import file_loadable_field


@dataclass
class OptimizationResult:
    """The result of a global analysis."""

    number_of_function_evaluations: int
    """The number of function evaluations."""

    success: bool
    """Indicates if the optimization was successful."""

    termination_reason: str
    """The reason (message when) the optimizer terminated"""

    glotaran_version: str
    """The glotaran version used to create the result."""

    free_parameter_labels: list[str]
    """List of labels of the free parameters used in optimization."""

    parameter_history: ParameterHistory = file_loadable_field(  # type:ignore[type-var]
        ParameterHistory
    )
    """The parameter history."""

    optimization_history: OptimizationHistory = file_loadable_field(  # type:ignore[type-var]
        OptimizationHistory
    )
    """The optimization history."""

    # The below can be none in case of unsuccessful optimization

    cost: ArrayLike | None = exclude_from_dict_field(None)
    """The final cost."""

    chi_square: float | None = None
    r"""The chi-square of the optimization.

    :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`."""

    covariance_matrix: ArrayLike | None = exclude_from_dict_field(None)
    """Covariance matrix.

    The rows and columns are corresponding to :attr:`free_parameter_labels`."""

    number_clp: int | None = None
    degrees_of_freedom: int | None = None
    """Degrees of freedom in optimization :math:`N - N_{vars}`."""

    jacobian: ArrayLike | list | None = exclude_from_dict_field(None)
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
        penalty: np.typing.ArrayLike,
        free_parameter_labels: list[str],
        termination_reason: str,
        number_clp: int,
    ):
        success = result is not None

        result_args = {
            "success": success,
            "glotaran_version": glotaran_version,
            "free_parameter_labels": free_parameter_labels,
            "parameter_history": parameter_history,
            "termination_reason": termination_reason,
            "optimization_history": optimization_history,
            "number_of_function_evaluations": result.nfev
            if success
            else parameter_history.number_of_records,
        }

        result_args["cost"] = 0.5 * np.dot(penalty, penalty)
        if success:
            result_args["number_clp"] = number_clp
            result_args["number_of_jacobian_evaluations"] = result.njev
            result_args["optimality"] = float(result.optimality)
            result_args["number_of_data_points"] = result.fun.size
            result_args["number_of_parameters"] = result.x.size
            result_args["degrees_of_freedom"] = (
                result_args["number_of_data_points"]
                - result_args["number_of_parameters"]
                - result_args["number_clp"]
            )
            result_args["chi_square"] = float(np.sum(result.fun**2))
            result_args["reduced_chi_square"] = (
                result_args["chi_square"] / result_args["degrees_of_freedom"]
            )
            result_args["root_mean_square_error"] = float(
                np.sqrt(result_args["reduced_chi_square"])
            )
            result_args["jacobian"] = result.jac
            result_args["covariance_matrix"] = calculate_covariance_matrix_and_standard_errors(
                result_args["jacobian"], result_args["root_mean_square_error"]
            )

        return cls(**result_args)

    def calculate_parameter_errors(self, parameters: Parameters):
        standard_errors = self.root_mean_square_error * np.sqrt(np.diag(self.covariance_matrix))
        for label, error in zip(self.free_parameter_labels, standard_errors):
            self._parameters.get(label).standard_error = error


def calculate_covariance_matrix_and_standard_errors(
    jacobian: np.typing.ArrayLike, root_mean_square_error: float
) -> np.typing.ArrayLike:
    """Calculate the covariance matrix and standard errors of the optimization.

    Parameters
    ----------
    jacobian : np.typing.ArrayLike
        The jacobian matrix.
    root_mean_square_error : float
        The root mean square error.

    Returns
    -------
    np.typing.ArrayLike
        The covariance matrix.
    """
    # See PR #706: More robust covariance matrix calculation
    _, jacobian_sv, jacobian_rsv = np.linalg.svd(jacobian, full_matrices=False)
    jacobian_sv_square = jacobian_sv**2
    mask = jacobian_sv_square > np.finfo(float).eps
    covariance_matrix = (jacobian_rsv[mask].T / jacobian_sv_square[mask]) @ jacobian_rsv[mask]
    return covariance_matrix
