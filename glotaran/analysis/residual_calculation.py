import dask
import dask.array as da
import dask.bag as db
import numpy as np


def create_index_independend_ungrouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    global_dimension = scheme.model.global_dimension
    clp_label = {}
    clps = {}
    residuals = {}
    full_residuals = []
    for label in problem_bag:
        data = problem_bag[label].data
        size = problem_bag[label].global_axis.size
        clp_label[label] = matrix_jobs[label][0]
        matrix = matrix_jobs[label][1]

        clps[label] = []
        residuals[label] = []
        for i in range(size):
            clp, residual = \
                dask.delayed(residual_function, nout=2)(matrix, data.isel({global_dimension: i}).values)
            clps[label].append(clp)
            residuals[label].append(residual)
            full_residuals.append(residual)

    full_residuals = dask.delayed(np.concatenate)(full_residuals)
    return clp_label, clps, residuals, full_residuals


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


    def residual(matrix, data, jobs):
        return residual_function(jobs[matrix][1], data)

    residual_bag = db.map(residual, matrices, data_bag, matrix_jobs)
    clps = residual_bag.pluck(0)
    full_residuals = dask.delayed(np.concatenate)(residual_bag.pluck(1))
    return clps, residual_bag.pluck(1), full_residuals


def create_index_dependend_grouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    data_bag = problem_bag.pluck(0)
    matrices = matrix_jobs.pluck(1)

    residual_bag = db.map(residual_function, matrices, data_bag)
    clps = residual_bag.pluck(0)
    full_residuals = dask.delayed(np.concatenate)(residual_bag.pluck(1))
    return clps, residual_bag, full_residuals
