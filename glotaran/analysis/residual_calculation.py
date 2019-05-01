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
        matrix = matrix_jobs[label].pluck(1)
        data_bag = db.from_sequence(data.isel({global_dimension: i}) for i in range(size))
        residual_bag = db.map(residual_function, data_bag, matrix)

        clps[label] = residual_bag.pluck(0)
        residuals[label] = residual_bag.pluck(1)
        full_residuals.append(residual_bag.pluck(1))
    full_residuals = db.concat(full_residuals).reduce(np.concatenate, np.concatenate)
    return clps, residuals, full_residuals


def create_index_independend_grouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    data_bag = problem_bag.pluck(0)
    datasets = problem_bag.pluck(1).map(lambda ps: [p.dataset for p in ps]).compute()

    labels = []
    for dataset in datasets:
        label = "".join(dataset)
        labels.append(label)
        if len(dataset) > 1:
            if label not in matrix_jobs:
                matrix_jobs[label] = _combine_matrices(
                    [matrix_jobs[d] for d in dataset]
                )

    matrices = db.from_sequence(matrix_jobs[label] for label in labels).pluck(1)

    residual_bag = db.map(residual_function, data_bag, matrices)
    clps = residual_bag.pluck(0)
    full_residuals = residual_bag.pluck(1).reduce(np.concatenate, np.concatenate)
    return clps, residual_bag, full_residuals


def create_index_dependend_grouped_residual(
        scheme, problem_bag, matrix_jobs, residual_function):

    data_bag = problem_bag.pluck(0)
    matrices = matrix_jobs.map(_combine_matrices).pluck(1)

    residual_bag = db.map(residual_function, data_bag, matrices)
    clps = residual_bag.pluck(0)
    full_residuals = residual_bag.pluck(1).reduce(np.concatenate, np.concatenate)
    return clps, residual_bag, full_residuals


def _combine_matrices(label_and_matrices):
    (all_clp, matrices) = label_and_matrices
    masks = []
    full_clp = None
    for clp in all_clp:
        if full_clp is None:
            full_clp = clp
            masks.append([i for i, _ in enumerate(clp)])
        else:
            mask = []
            for c in clp:
                if c not in full_clp:
                    full_clp.append(c)
                mask.append(full_clp.index(c))
            masks.append(mask)
    dim1 = np.sum([m.shape[0] for m in matrices])
    dim2 = len(full_clp)
    matrix = np.zeros((dim1, dim2), dtype=np.float64)
    start = 0
    for i, m in enumerate(matrices):
        end = start + m.shape[0]
        matrix[start:end, masks[i]] = m
        start = end

    return LabelAndMatrix(full_clp, matrix)
