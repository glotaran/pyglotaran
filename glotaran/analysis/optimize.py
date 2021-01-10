from typing import List

import numpy as np
from scipy.optimize import least_squares

from .problem import Problem
from .result import Result
from .scheme import Scheme

SUPPORTED_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "LevenbergMarquart": "lm",
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
    ) = problem.scheme.parameter.get_label_value_and_bounds_arrays(exclude_non_vary=True)
    method = SUPPORTED_METHODS[problem.scheme.optimization_method]
    nfev = problem.scheme.nfev
    verbose = 2 if verbose else 0

    ls_result = least_squares(
        _calculate_penalty,
        initial_parameter,
        bounds=(lower_bounds, upper_bounds),
        method=method,
        max_nfev=nfev,
        verbose=verbose,
        #  ftol=1.e-7,
        #  gtol=1.e-7,
        kwargs={"labels": labels, "problem": problem},
    )

    return Result(
        problem.scheme,
        problem.create_result_data(),
        problem.parameter,
        ls_result,
        labels,
    )


def _calculate_penalty(parameter: np.ndarray, labels: List[str] = None, problem: Problem = None):
    problem.parameter.set_from_label_and_value_arrays(labels, parameter)
    problem.reset()
    return problem.full_penalty
