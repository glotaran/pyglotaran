import dask
import dask.bag as db
import numpy as np


def create_index_independend_ungrouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    global_dimension = scheme.model.global_dimension
    clps = {}
    residuals = {}
    full_residuals = []
    for label in problem_bag:
        data = problem_bag[label].data
        size = problem_bag[label].global_axis.size
        matrix = matrix_jobs[label].matrix

        clps[label] = []
        residuals[label] = []
        for i in range(size):
            clp, residual = \
                dask.delayed(residual_function, nout=2)(matrix, data.isel({global_dimension: i}))
            clps[label].append(clp)
            residuals[label].append(residual)
            full_residuals.append(residual)

    full_residuals = dask.delayed(np.concatenate)(full_residuals)
    return clps, residuals, full_residuals


def create_index_dependend_ungrouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    global_dimension = scheme.model.global_dimension
    clps = {}
    residuals = {}
    full_residuals = []
    for label in problem_bag:
        data = problem_bag[label].data
        size = problem_bag[label].global_axis.size
        matrices = matrix_jobs[label]
        clps[label] = []
        residuals[label] = []
        for i in range(size):
            clp, residual = \
                dask.delayed(residual_function, nout=2)(
                    matrices[i][1], data.isel({global_dimension: i}))
            clps[label].append(clp)
            residuals[label].append(residual)
            full_residuals.append(residual)
    full_residuals = db.concat(full_residuals)
    return clps, residuals, full_residuals


def create_index_independend_grouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    data_bag = problem_bag.pluck(0)

    matrices = problem_bag.pluck(1)\
        .map(lambda group: "".join(problem.dataset for problem in group))\
        .map(lambda label, jobs: jobs[label], matrix_jobs)\
        .pluck(1)

    residual_bag = db.map(lambda mat, data: residual_function(mat, data), matrices, data_bag)
    clps = residual_bag.pluck(0)
    full_residuals = db.concat(residual_bag.pluck(1))
    return clps, residual_bag.pluck(1), full_residuals


def create_index_dependend_grouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    data_bag = problem_bag.pluck(0)
    matrices = matrix_jobs.map(_combine_matrices).pluck(1)

    residual_bag = db.map(residual_function, data_bag, matrices)
    clps = residual_bag.pluck(0)
    full_residuals = residual_bag.pluck(1).reduce(np.concatenate, np.concatenate)
    return clps, residual_bag, full_residuals
