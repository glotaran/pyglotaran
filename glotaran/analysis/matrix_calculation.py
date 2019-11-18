import collections
import dask
import numpy as np


LabelAndMatrix = collections.namedtuple('LabelAndMatrix', 'clp_label matrix')
LabelAndMatrixAndData = collections.namedtuple('LabelAndMatrixAndData', 'label_matrix data')


def calculate_index_independend_ungrouped_matrices(scheme, parameter):

    # for this we don't use dask

    clp_labels = {}
    matrices = {}
    constraint_labels_and_matrices = {}
    model = scheme.model

    descriptors = {label: descriptor.fill(model, parameter)
                   for label, descriptor in scheme.model.dataset.items()}

    for label, descriptor in descriptors.items():
        axis = scheme.data[label].coords[model.matrix_dimension].values
        clp_label, matrix = _calculate_matrix(
            model.matrix,
            descriptor,
            axis,
            {},
        )
        clp_labels[label] = clp_label
        matrices[label] = matrix

        if callable(model.has_matrix_constraints_function):
            if model.has_matrix_constraints_function():
                clp_label, matrix = \
                    model.constrain_matrix_function(parameter, clp_label, matrix, None)

        constraint_labels_and_matrices[label] = LabelAndMatrix(clp_label, matrix)
    return clp_labels, matrices, constraint_labels_and_matrices


def calculate_index_independend_grouped_matrices(scheme, groups, parameter):

    # We just need to create groups from the ungrouped matrices
    clp_labels, matrices, constraint_labels_and_matrices = \
        calculate_index_independend_ungrouped_matrices(scheme, parameter)
    for label, group in groups.items():
        if label not in matrices:
            constraint_labels_and_matrices[label] = \
                _combine_matrices([constraint_labels_and_matrices[l] for l in group])

    return clp_labels, matrices, constraint_labels_and_matrices


def create_index_dependend_ungrouped_matrix_jobs(scheme, bag, parameter):

    model = scheme.model
    clp_labels = {}
    matrices = {}
    constraint_labels_and_matrices = {}

    descriptors = {label: descriptor.fill(model, parameter)
                   for label, descriptor in scheme.model.dataset.items()}
    for label, problem in bag.items():
        descriptor = descriptors[label]
        clp_labels[label] = []
        matrices[label] = []
        constraint_labels_and_matrices[label] = []
        for index in problem.global_axis:
            clp, matrix = dask.delayed(_calculate_matrix, nout=2)(
                model.matrix,
                descriptor,
                problem.matrix_axis,
                {},
                index=index,
            )
            clp_labels[label].append(clp)
            matrices[label].append(matrix)

            if callable(model.has_matrix_constraints_function):
                if model.has_matrix_constraints_function():
                    clp, matrix = dask.delayed(model.constrain_matrix_function, nout=2)(
                        parameter, clp, matrix, index
                    )
            constraint_labels_and_matrices[label].append((clp, matrix))

    return clp_labels, matrices, constraint_labels_and_matrices


def create_index_dependend_grouped_matrix_jobs(scheme, bag, parameter):

    model = scheme.model

    descriptors = {label: descriptor.fill(model, parameter)
                   for label, descriptor in scheme.model.dataset.items()}

    def calculate_group(group):
        results = [_calculate_matrix(
            model.matrix,
            descriptors[problem.dataset],
            problem.axis,
            {},
            index=problem.index
        ) for problem in group.descriptor]
        return results, group.descriptor[0].index

    def get_clp(result):
        return [d[0] for d in result[0]]

    def get_matrices(result):
        return [d[1] for d in result[0]]

    def constrain_and_combine_matrices(result):
        matrices, index = result
        clp, matrix = _combine_matrices(matrices)
        if callable(model.has_matrix_constraints_function):
            if model.has_matrix_constraints_function():
                clp, matrix = model.constrain_matrix_function(parameter, clp, matrix, index)
        return LabelAndMatrix(clp, matrix)

    matrix_jobs = bag.map(calculate_group)
    constraint_matrix_jobs = matrix_jobs.map(constrain_and_combine_matrices)
    clp = matrix_jobs.map(get_clp)
    matrices = matrix_jobs.map(get_matrices)
    return clp, matrices, constraint_matrix_jobs


def _calculate_matrix(matrix_function, dataset_descriptor, axis, extra, index=None):
    args = {
        'dataset_descriptor': dataset_descriptor,
        'axis': axis,
    }
    for k, v in extra:
        args[k] = v
    if index is not None:
        args['index'] = index
    clp_label, matrix = matrix_function(**args)
    if dataset_descriptor.scale is not None:
        matrix *= dataset_descriptor.scale
    return LabelAndMatrix(clp_label, matrix)


def _combine_matrices(labels_and_matrices):
    masks = []
    full_clp_labels = None
    sizes = []
    for label_and_matrix in labels_and_matrices:
        (clp_label, matrix) = label_and_matrix
        sizes.append(matrix.shape[0])
        if full_clp_labels is None:
            full_clp_labels = clp_label
            masks.append([i for i, _ in enumerate(clp_label)])
        else:
            mask = []
            for c in clp_label:
                if c not in full_clp_labels:
                    full_clp_labels.append(c)
                mask.append(full_clp_labels.index(c))
            masks.append(mask)
    dim1 = np.sum(sizes)
    dim2 = len(full_clp_labels)
    full_matrix = np.zeros((dim1, dim2), dtype=np.float64)
    start = 0
    for i, m in enumerate(labels_and_matrices):
        end = start + sizes[i]
        full_matrix[start:end, masks[i]] = m[1]
        start = end

    return (full_clp_labels, full_matrix)
