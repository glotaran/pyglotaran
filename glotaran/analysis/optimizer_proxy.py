import collections
import dask.distributed as dd
import lmfit

from . import problem_bag, matrix_calculation, residual_calculation
from .parameter_server import ParameterClient
from .nnls import residual_nnls
from .variable_projection import residual_variable_projection

ResultFuture = collections.namedtuple('ResultFuture', 'clp_label matrix clp residual')


def optimize(scheme, verbose=True):

    parameter = scheme.parameter.as_parameter_dict()
    residual_function = residual_nnls if scheme.nnls else residual_variable_projection
    client = dd.get_client()
    with ParameterClient(client) as parameter_client:
        result, penalty = create_index_independent_ungrouped_problem(
            scheme, parameter_client, residual_function
        )

        minimizer = lmfit.Minimizer(
            calculate_penalty,
            parameter,
            fcn_args=[parameter_client, penalty],
            fcn_kws=None,
            iter_cb=None,
            scale_covar=True,
            nan_policy='omit',
            reduce_fcn=None,
            **{})
        verbose = 2 if verbose else 0
        lm_result = minimizer.minimize(
            method='least_squares', verbose=verbose, max_nfev=scheme.nfev)


def calculate_penalty(parameter, parameter_client, penalty_job):
    parameter_client.update(parameter).result()
    return penalty_job.compute()


def create_index_independent_ungrouped_problem(scheme, parameter_client, residual_function):

    bag = problem_bag.create_ungrouped_bag(scheme)

    matrix_jobs = \
        matrix_calculation.create_index_independend_matrix_jobs(scheme, parameter_client)

    residual_job = residual_calculation.create_index_independend_ungrouped_residual(
        scheme, bag, matrix_jobs, residual_function)

    results = {}
    for label in scheme.model.dataset:
        results[label] = ResultFuture(
            matrix_jobs[label].pluck(0),
            matrix_jobs[label].pluck(1),
            residual_job[0][label],
            residual_job[1][label],
        )
    return results, residual_job[2]
