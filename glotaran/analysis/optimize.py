from __future__ import annotations

from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran.analysis.problem import Problem
from glotaran.analysis.problem_grouped import GroupedProblem
from glotaran.analysis.problem_ungrouped import UngroupedProblem
from glotaran.project import Result
from glotaran.project import Scheme

SUPPORTED_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


def optimize(scheme: Scheme, verbose: bool = True) -> Result:
    problem = GroupedProblem(scheme) if scheme.model.grouped() else UngroupedProblem(scheme)
    return optimize_problem(problem, verbose=verbose)


def optimize_problem(problem: Problem, verbose: bool = True) -> Result:

    if problem.scheme.optimization_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported optimization method {problem.scheme.optimization_method}. "
            f"Supported methods are '{list(SUPPORTED_METHODS.keys())}'"
        )

    (
        free_parameter_labels,
        initial_parameter,
        lower_bounds,
        upper_bounds,
    ) = problem.scheme.parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)
    method = SUPPORTED_METHODS[problem.scheme.optimization_method]
    nfev = problem.scheme.maximum_number_function_evaluations
    ftol = problem.scheme.ftol
    gtol = problem.scheme.gtol
    xtol = problem.scheme.xtol
    verbose = 2 if verbose else 0
    termination_reason = ""

    try:
        ls_result = least_squares(
            _calculate_penalty,
            initial_parameter,
            bounds=(lower_bounds, upper_bounds),
            method=method,
            max_nfev=nfev,
            verbose=verbose,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            kwargs={"free_parameter_labels": free_parameter_labels, "problem": problem},
        )
        termination_reason = ls_result.message
    except Exception as e:
        warn(f"Optimization failed:\n\n{e}")
        termination_reason = str(e)
        ls_result = None

    problem.save_parameters_for_history()

    return _create_result(problem, ls_result, free_parameter_labels, termination_reason)


def _calculate_penalty(
    parameters: np.ndarray, free_parameter_labels: list[str] = None, problem: Problem = None
):
    problem.save_parameters_for_history()
    problem.parameters.set_from_label_and_value_arrays(free_parameter_labels, parameters)
    problem.reset()
    return problem.full_penalty


def _create_result(
    problem: Problem,
    ls_result: OptimizeResult | None,
    free_parameter_labels: list[str],
    termination_reason: str,
) -> Result:

    success = ls_result is not None

    number_of_function_evaluation = (
        ls_result.nfev if ls_result is not None else len(problem.parameter_history)
    )
    number_of_jacobian_evaluation = ls_result.njev if success else None
    optimality = ls_result.optimality if success else None
    number_of_data_points = ls_result.fun.size if success else None
    number_of_variables = ls_result.x.size if success else None
    degrees_of_freedom = number_of_data_points - number_of_variables if success else None
    chi_square = np.sum(ls_result.fun ** 2) if success else None
    reduced_chi_square = chi_square / degrees_of_freedom if success else None
    root_mean_square_error = np.sqrt(reduced_chi_square) if success else None
    jacobian = ls_result.jac if success else None

    problem.save_parameters_for_history()
    history_index = None if success else -2
    data = problem.create_result_data(history_index=history_index)
    # the optimized parameters are those of the last run if the optimization has crashed
    parameters = problem.parameters
    covariance_matrix = None
    if success:
        # See PR #706: More robust covariance matrix calculation
        _, jacobian_SV, jacobian_RSV = np.linalg.svd(jacobian, full_matrices=False)
        jacobian_SV_square = jacobian_SV ** 2
        mask = jacobian_SV_square > np.finfo(float).eps
        covariance_matrix = (jacobian_RSV[mask].T / jacobian_SV_square[mask]) @ jacobian_RSV[mask]
        standard_errors = root_mean_square_error * np.sqrt(np.diag(covariance_matrix))
        for label, error in zip(free_parameter_labels, standard_errors):
            parameters.get(label).standard_error = error

    return Result(
        additional_penalty=problem.additional_penalty,
        cost=problem.cost,
        data=data,
        free_parameter_labels=free_parameter_labels,
        number_of_function_evaluations=number_of_function_evaluation,
        initial_parameters=problem.scheme.parameters,
        optimized_parameters=parameters,
        scheme=problem.scheme,
        success=success,
        termination_reason=termination_reason,
        chi_square=chi_square,
        covariance_matrix=covariance_matrix,
        degrees_of_freedom=degrees_of_freedom,
        jacobian=jacobian,
        number_of_data_points=number_of_data_points,
        number_of_jacobian_evaluations=number_of_jacobian_evaluation,
        number_of_variables=number_of_variables,
        optimality=optimality,
        reduced_chi_square=reduced_chi_square,
        root_mean_square_error=root_mean_square_error,
    )
