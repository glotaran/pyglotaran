import collections
import dask
import dask.bag as db

from glotaran.parameter import ParameterGroup

LabelAndMatrix = collections.namedtuple('LabelAndMatrix', 'clp_label matrix')
LabelAndMatrixAndData = collections.namedtuple('LabelAndMatrixAndData', 'label_matrix data')


def create_index_independend_matrix_jobs(scheme, parameter_client):

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


def create_index_dependend_ungrouped_matrix_jobs(scheme, bag, parameter_client):

    model = scheme.model
    matrix_jobs = {}

    for label, problem in bag.items():
        matrix_bag = db.from_sequence(problem.global_axis)
        descriptor = _fill_dataset_descriptor(model, problem.dataset, parameter_client)

        matrix_bag.map(lambda index: _calculate_matrix(
            model.matrix_function,
            descriptor,
            problem.matrix_axis,
            {},
            index=index,
        ))
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
    return LabelAndMatrix(clp_label, matrix)
