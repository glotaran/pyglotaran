from typing import List
from warnings import warn

import numpy as np
from scipy.optimize import least_squares

from .problem import Problem
from .result import Result
from .scheme import Scheme

SUPPORTED_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


def optimize(scheme: Scheme, verbose: bool = True) -> Result:
    problem = Problem(scheme)
    return optimize_problem(problem, verbose=verbose)


def optimize_problem(problem: Problem, verbose: bool = True) -> Result:

    if problem.scheme.optimization_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported optimization method {problem.scheme.optimization_method}. "
            f"Supported methods are '{list(SUPPORTED_METHODS.keys())}'"
        )

    (
        labels,
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
    history_index = -1

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
            kwargs={"labels": labels, "problem": problem},
        )
        termination_reason = ls_result.message
    except Exception as e:
        warn(f"Optimization failed:\n\n{e}")
        termination_reason = str(e)
        ls_result = None
        history_index = -2

    problem.save_parameters_for_history()

    return Result(
        problem.scheme,
        problem.create_result_data(history_index=history_index),
        problem.parameters,
        problem.additional_penalty,
        ls_result,
        labels,
        termination_reason,
    )


def _calculate_penalty(parameters: np.ndarray, labels: List[str] = None, problem: Problem = None):
    problem.save_parameters_for_history()
    problem.parameters.set_from_label_and_value_arrays(labels, parameters)
    problem.reset()
    return problem.full_penalty
