import collections
import dask
import dask.bag as db
import numpy as np

from glotaran.parameter import ParameterGroup

LabelAndMatrix = collections.namedtuple('LabelAndMatrix', 'clp_label matrix')
LabelAndMatrixAndData = collections.namedtuple('LabelAndMatrixAndData', 'label_matrix data')


def create_index_independend_ungrouped_matrix_jobs(scheme, parameter_client):

    matrix_jobs = {}
    model = scheme.model

    for label, descriptor in scheme.model.dataset.items():
        descriptor = _fill_dataset_descriptor(model, descriptor, parameter_client)
        matrix_jobs[label] = dask.delayed(_calculate_matrix)(
            model.matrix,
            descriptor,
            scheme.data[label].coords[model.matrix_dimension],
            {},
        )
    return matrix_jobs


def create_index_independend_grouped_matrix_jobs(scheme, groups, parameter_client):

    matrix_jobs = create_index_independend_ungrouped_matrix_jobs(scheme, parameter_client)

    for label, group in groups.items():
        matrix_jobs[label] = dask.delayed(_combine_matrices, nout=2)(
                    [matrix_jobs[d] for d in group]
                )

    return matrix_jobs


def create_index_dependend_ungrouped_matrix_jobs(scheme, bag, parameter_client):

    model = scheme.model
    matrix_jobs = {}

    for label, problem in bag.items():
        descriptor = _fill_dataset_descriptor(model, problem.dataset, parameter_client)

        matrix_bag = [dask.delayed(_calculate_matrix, nout=2)(
            model.matrix,
            descriptor,
            problem.matrix_axis,
            {},
            index=index,
        ) for index in problem.global_axis.values]
        matrix_jobs[label] = matrix_bag

    return matrix_jobs


def create_index_dependend_grouped_matrix_jobs(scheme, bag, parameter_client):

    model = scheme.model

    descriptors = {label: _fill_dataset_descriptor(descriptor)
                   for label, descriptor in model.dataset}

    def calculate_group(group):
        return [_calculate_matrix(
            model.matrix_function,
            descriptors[problem.dataset],
            problem.axis,
            {},
            index=problem.index
        ) for problem in group]

    matrix_jobs = bag.map(calculate_group)
    return matrix_jobs


@dask.delayed
def _fill_dataset_descriptor(model, descriptor, parameter_client):
    parameter = parameter_client.get().result()
    parameter = ParameterGroup.from_parameter_dict(parameter)
    return descriptor.fill(model, parameter)


def _calculate_matrix(matrix_function, dataset_descriptor, axis, extra, index=None):
    args = {
        'dataset_descriptor': dataset_descriptor,
        'axis': axis.values,
    }
    for k, v in extra:
        args[k] = v
    if index is not None:
        args['index'] = index
    clp_label, matrix = matrix_function(**args)
    if dataset_descriptor.scale is not None:
        matrix *= dataset_descriptor.scale
    return clp_label, matrix


def _combine_matrices(label_and_matrices):
    (all_clp, matrices) = ([], [])
    masks = []
    full_clp = None
    for label_and_matrix in label_and_matrices:
        (clp, matrix) = label_and_matrix
        matrices.append(matrix)
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

    return (full_clp, matrix)
