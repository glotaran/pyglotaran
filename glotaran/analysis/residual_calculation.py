import dask
import numpy as np
from dask import bag as db


def create_index_independent_ungrouped_residual(
    scheme, parameter, problem_bag, constraint_labels_and_matrices, residual_function
):

    global_dimension = scheme.model.global_dimension
    reduced_clp_labels = {}
    reduced_clps = {}
    residuals = {}
    penalties = []
    for label in problem_bag:
        data = problem_bag[label].data
        weight = problem_bag[label].weight
        global_axis = problem_bag[label].global_axis
        reduced_clp_labels[label] = constraint_labels_and_matrices[label].clp_label
        matrix = constraint_labels_and_matrices[label].matrix

        reduced_clps[label] = []
        residuals[label] = []
        for i, index in enumerate(global_axis):
            data_stripe = data.isel({global_dimension: i}).values
            matrix_stripe = matrix

            if weight is not None:
                for j in range(matrix.shape[1]):
                    matrix[:, j] *= weight.isel({global_dimension: i}).values

            clp, residual = dask.delayed(residual_function, nout=2)(matrix_stripe, data_stripe)
            reduced_clps[label].append(clp)
            residuals[label].append(residual)
            penalties.append(residual)

        _apply_ungrouped_penalty(
            scheme, parameter, label, reduced_clp_labels, reduced_clps, global_axis
        )

    penalty = dask.delayed(np.concatenate)(penalties)
    return reduced_clp_labels, reduced_clps, residuals, penalty


def create_index_dependent_ungrouped_residual(
    scheme, parameter, problem_bag, matrix_jobs, residual_function
):
    def apply_weight(matrix, weight):
        for i in range(matrix.shape[1]):
            matrix[:, i] *= weight
        return matrix

    global_dimension = scheme.model.global_dimension
    reduced_clp_labels = {}
    reduced_clps = {}
    residuals = {}
    penalties = []
    for label in problem_bag:
        data = problem_bag[label].data
        global_axis = problem_bag[label].global_axis
        matrices = matrix_jobs[label]
        weight = problem_bag[label].weight
        reduced_clp_labels[label] = []
        reduced_clps[label] = []
        residuals[label] = []
        for i, index in enumerate(global_axis):
            matrix = matrices[i][1]
            if weight is not None:
                matrix = dask.delayed(apply_weight)(matrix, weight.isel({global_dimension: i}))
            clp, residual = dask.delayed(residual_function, nout=2)(
                matrix, data.isel({global_dimension: i}).values
            )

            clp_label = matrices[i][0]
            reduced_clp_labels[label].append(clp_label)
            reduced_clps[label].append(clp)
            residuals[label].append(residual)
            penalties.append(residual)

        _apply_ungrouped_penalty(
            scheme, parameter, label, reduced_clp_labels, reduced_clps, global_axis
        )

    penalty = dask.delayed(np.concatenate)(penalties)
    return reduced_clp_labels, reduced_clps, residuals, penalty


def create_index_independent_grouped_residual(
    scheme, parameter, problem_bag, constraint_labels_and_matrices, residual_function
):

    matrix_labels = problem_bag.pluck(2).map(
        lambda group: "".join(problem.dataset for problem in group)
    )

    def penalty_function(matrix_label, problem, labels_and_matrices):

        matrix = labels_and_matrices[matrix_label].matrix
        for i in range(matrix.shape[1]):
            matrix[:, i] *= problem.weight
        clp, residual = residual_function(matrix, problem.data)

        return clp, residual

    penalty_bag = db.map(
        penalty_function, matrix_labels, problem_bag, constraint_labels_and_matrices
    )

    reduced_clp_labels = {
        label: constraint_labels_and_matrices[label].clp_label
        for label in constraint_labels_and_matrices
    }
    reduced_clps = penalty_bag.pluck(0)
    residuals = penalty_bag.pluck(1)
    penalty = _apply_grouped_penalty(
        scheme, parameter, problem_bag, reduced_clp_labels, reduced_clps, residuals
    )

    return reduced_clp_labels, reduced_clps, residuals, penalty


def create_index_dependent_grouped_residual(
    scheme, parameter, problem_bag, constraint_labels_and_matrices, residual_function
):
    def penalty_function(problem, labels_and_matrices):
        matrix = labels_and_matrices.matrix
        for i in range(matrix.shape[1]):
            matrix[:, i] *= problem.weight

        clp, residual = residual_function(matrix, problem.data)

        return clp, residual

    penalty_bag = db.map(penalty_function, problem_bag, constraint_labels_and_matrices)

    reduced_clp_labels = constraint_labels_and_matrices.pluck(0)
    reduced_clps = penalty_bag.pluck(0)
    residuals = penalty_bag.pluck(1)
    penalty = _apply_grouped_penalty(
        scheme, parameter, problem_bag, reduced_clp_labels, reduced_clps, residuals
    )

    return reduced_clp_labels, reduced_clps, residuals, penalty


def _apply_grouped_penalty(
    scheme, parameter, problem_bag, reduced_clp_labels, reduced_clps, penalties
):
    if (
        callable(scheme.model.has_additional_penalty_function)
        and scheme.model.has_additional_penalty_function()
    ):
        full_axis = problem_bag.pluck(2).map(lambda group: group[0].index)

        additional_penalty = scheme.model.additional_penalty_function(
            parameter, reduced_clp_labels, reduced_clps, full_axis
        )
        penalties = dask.delayed(np.concatenate)([penalties, additional_penalty])
    return penalties


def _apply_ungrouped_penalty(
    scheme, parameter, label, reduced_clp_labels, reduced_clps, global_axis, penalties
):
    if (
        callable(scheme.model.has_additional_penalty_function)
        and scheme.model.has_additional_penalty_function()
    ):
        additional_penalty = dask.delayed(scheme.model.additional_penalty_function)(
            parameter, reduced_clp_labels[label], reduced_clps[label], global_axis
        )
        penalties.append(additional_penalty)
