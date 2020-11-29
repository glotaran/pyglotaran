import lmfit

from glotaran.parameter import ParameterGroup

from .problem import Problem
from .result import Result
from .scheme import Scheme


def optimize(scheme: Scheme, verbose: bool = True) -> Result:

    initial_parameter = scheme.parameter.as_parameter_dict()
    problem = Problem(scheme)

    minimizer = lmfit.Minimizer(
        _calculate_penalty,
        initial_parameter,
        fcn_args=[problem],
        fcn_kws=None,
        iter_cb=None,
        scale_covar=True,
        nan_policy="omit",
        reduce_fcn=None,
        **{},
    )
    verbose = 2 if verbose else 0
    lm_result = minimizer.minimize(method="least_squares", verbose=verbose, max_nfev=scheme.nfev)

    covar = lm_result.covar if hasattr(lm_result, "covar") else None

    return Result(
        scheme,
        problem.create_result_data(),
        problem.parameter,
        lm_result.nfev,
        lm_result.nvarys,
        lm_result.ndata,
        lm_result.nfree,
        lm_result.chisqr,
        lm_result.redchi,
        lm_result.var_names,
        covar,
    )


def _calculate_penalty(parameter: lmfit.Parameters, problem: Problem):
    problem.parameter = ParameterGroup.from_parameter_dict(parameter)
    return problem.full_penalty
