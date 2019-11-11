import dask
import dask.bag as db
import numpy as np


def create_index_independend_ungrouped_residual(
        scheme, parameter, problem_bag, constraint_labels_and_matrices, residual_function):

    global_dimension = scheme.model.global_dimension
    reduced_clp_labels = {}
    reduced_clps = {}
    residuals = {}
    penalties = []
    for label in problem_bag:
        data = problem_bag[label].data
        size = problem_bag[label].global_axis.size
        reduced_clp_labels[label] = constraint_labels_and_matrices[label].clp_label
        matrix = constraint_labels_and_matrices[label].matrix

        reduced_clps[label] = []
        residuals[label] = []
        for i in range(size):
            data_stripe = data.isel({global_dimension: i}).values
            clp, residual = \
                dask.delayed(residual_function, nout=2)(matrix, data_stripe)
            reduced_clps[label].append(clp)
            residuals[label].append(residual)
            penalties.append(residual)

            if callable(scheme.model.has_additional_penalty_function):
                if scheme.model.has_additional_penalty_function():
                    additional_penalty = dask.delayed(scheme.model.additional_penalty_function)(
                        parameter, reduced_clp_labels[label], reduced_clps[label], i
                    )
                    penalties.append(additional_penalty)

    penalty = dask.delayed(np.concatenate)(penalties)
    return reduced_clp_labels, reduced_clps, residuals, penalty


def create_index_dependend_ungrouped_residual(
        scheme, parameter, problem_bag, matrix_jobs, residual_function):

    global_dimension = scheme.model.global_dimension
    reduced_clp_labels = {}
    reduced_clps = {}
    residuals = {}
    penalties = []
    for label in problem_bag:
        data = problem_bag[label].data
        size = problem_bag[label].global_axis.size
        matrices = matrix_jobs[label]
        reduced_clp_labels[label] = []
        reduced_clps[label] = []
        residuals[label] = []
        for i in range(size):
            clp, residual = \
                dask.delayed(residual_function, nout=2)(
                    matrices[i][1], data.isel({global_dimension: i}).values)

            clp_label = matrices[i][0]
            reduced_clp_labels[label].append(clp_label)
            reduced_clps[label].append(clp)
            residuals[label].append(residual)
            penalties.append(residual)

            if callable(scheme.model.has_additional_penalty_function):
                if scheme.model.has_additional_penalty_function():
                    additional_penalty = dask.delayed(scheme.model.additional_penalty_function)(
                        parameter, clp_label, clp, i
                    )
                    penalties.append(additional_penalty)

    penalty = dask.delayed(np.concatenate)(penalties)
    return reduced_clp_labels, reduced_clps, residuals, penalty


def create_index_independend_grouped_residual(
        scheme, parameter, problem_bag, constraint_labels_and_matrices, residual_function):

    matrix_labels = problem_bag.pluck(1)\
        .map(lambda group: "".join(problem.dataset for problem in group))\

    def penalty_function(matrix_label, problem, labels_and_matrices):

        clp, residual = residual_function(labels_and_matrices[matrix_label].matrix, problem.data)

        penalty = residual
        if callable(scheme.model.has_additional_penalty_function):
            if scheme.model.has_additional_penalty_function():
                additional_penalty = scheme.model.additional_penalty_function(
                    parameter, labels_and_matrices[matrix_label].clp_label, clp, problem.index
                )
                penalty = np.concatenate([penalty, additional_penalty])
        return clp, residual, penalty
    penalty_bag = \
        db.map(penalty_function, matrix_labels, problem_bag, constraint_labels_and_matrices)

    reduced_clp_label = {label: constraint_labels_and_matrices[label].clp_label
                         for label in constraint_labels_and_matrices}
    reduced_clps = penalty_bag.pluck(0)
    residuals = penalty_bag.pluck(1)
    penalty = dask.delayed(np.concatenate)(penalty_bag.pluck(2))

    return reduced_clp_label, reduced_clps, residuals, penalty


def create_index_dependend_grouped_residual(
        scheme, parameter, problem_bag, constraint_labels_and_matrices, residual_function):

    def penalty_function(problem, labels_and_matrices):

        clp, residual = residual_function(labels_and_matrices.matrix, problem.data)

        penalty = residual
        if callable(scheme.model.has_additional_penalty_function):
            if scheme.model.has_additional_penalty_function():
                additional_penalty = scheme.model.additional_penalty_function(
                    parameter, labels_and_matrices.clp_label, clp, problem.index
                )
                penalty = np.concatenate([penalty, additional_penalty])
        return clp, residual, penalty
    penalty_bag = \
        db.map(penalty_function, problem_bag, constraint_labels_and_matrices)

    reduced_clp_labels = constraint_labels_and_matrices.pluck(0)
    reduced_clps = penalty_bag.pluck(0)
    residuals = penalty_bag.pluck(1)
    penalty = dask.delayed(np.concatenate)(penalty_bag.pluck(2))

    return reduced_clp_labels, reduced_clps, residuals, penalty
