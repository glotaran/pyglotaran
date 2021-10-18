from __future__ import annotations

from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran import __version__ as glotaran_version
from glotaran.analysis.optimization_group import OptimizationGroup
from glotaran.parameter import ParameterHistory
from glotaran.project import Result
from glotaran.project import Scheme

SUPPORTED_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


def optimize(scheme: Scheme, verbose: bool = True, raise_exception: bool = False) -> Result:

    optimization_groups = [
        OptimizationGroup(scheme, group) for group in scheme.model.get_dataset_groups().values()
    ]

    (
        free_parameter_labels,
        initial_parameter,
        lower_bounds,
        upper_bounds,
    ) = scheme.parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)

    if scheme.optimization_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported optimization method {scheme.optimization_method}. "
            f"Supported methods are '{list(SUPPORTED_METHODS.keys())}'"
        )
    method = SUPPORTED_METHODS[scheme.optimization_method]

    nfev = scheme.maximum_number_function_evaluations
    ftol = scheme.ftol
    gtol = scheme.gtol
    xtol = scheme.xtol
    verbose = 2 if verbose else 0
    termination_reason = ""

    parameter_history = ParameterHistory()
    parameter_history.append(scheme.parameters)
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
            kwargs={
                "free_parameter_labels": free_parameter_labels,
                "optimization_groups": optimization_groups,
                "parameter_history": parameter_history,
            },
        )
        termination_reason = ls_result.message
    except Exception as e:
        if raise_exception:
            raise e
        warn(f"Optimization failed:\n\n{e}")
        termination_reason = str(e)
        ls_result = None

    return _create_result(
        scheme,
        optimization_groups,
        ls_result,
        free_parameter_labels,
        termination_reason,
        parameter_history,
    )


def _calculate_penalty(
    parameters: np.ndarray,
    *,
    free_parameter_labels: list[str],
    optimization_groups: list[OptimizationGroup],
    parameter_history: ParameterHistory,
):
    for group in optimization_groups:
        group.parameters.set_from_label_and_value_arrays(free_parameter_labels, parameters)
        group.reset()
    parameter_history.append(
        optimization_groups[0].parameters
    )  # parameters are the same for all groups

    penalties = [group.full_penalty for group in optimization_groups]

    return np.concatenate(penalties) if len(penalties) != 1 else penalties[0]


def _create_result(
    scheme: Scheme,
    optimization_groups: list[OptimizationGroup],
    ls_result: OptimizeResult | None,
    free_parameter_labels: list[str],
    termination_reason: str,
    parameter_history: ParameterHistory,
) -> Result:

    success = ls_result is not None

    number_of_function_evaluation = (
        ls_result.nfev if success else parameter_history.number_of_records
    )
    number_of_jacobian_evaluation = ls_result.njev if success else None
    optimality = float(ls_result.optimality) if success else None
    number_of_data_points = ls_result.fun.size if success else None
    number_of_variables = ls_result.x.size if success else None
    degrees_of_freedom = number_of_data_points - number_of_variables if success else None
    chi_square = float(np.sum(ls_result.fun ** 2)) if success else None
    reduced_chi_square = chi_square / degrees_of_freedom if success else None
    root_mean_square_error = float(np.sqrt(reduced_chi_square)) if success else None
    jacobian = ls_result.jac if success else None

    if success:
        for group in optimization_groups:
            group.parameters.set_from_label_and_value_arrays(free_parameter_labels, ls_result.x)
            group.reset()

    data = {}
    for group in optimization_groups:
        data.update(
            group.create_result_data(parameter_history, success=success, add_svd=scheme.add_svd)
        )

    # the optimized parameters are those of the last run if the optimization has crashed
    parameters = optimization_groups[0].parameters
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

    additional_penalty = [group.additional_penalty for group in optimization_groups]

    cost = [group.cost for group in optimization_groups]

    return Result(
        additional_penalty=additional_penalty,
        cost=cost,
        data=data,
        glotaran_version=glotaran_version,
        free_parameter_labels=free_parameter_labels,
        number_of_function_evaluations=number_of_function_evaluation,
        initial_parameters=scheme.parameters,
        optimized_parameters=parameters,
        parameter_history=parameter_history,
        scheme=scheme,
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
