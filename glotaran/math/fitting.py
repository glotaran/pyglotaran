from lmfit import Minimizer

import numpy as np
from glotaran.model.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup
from .grouping import create_group, calculate_group, get_data_group
from .variable_projection import qr_residual, qr_coefficents
from .fitresult import DatasetResult, FitResult


def fit(model, parameter):

    group = create_group(model)
    data = get_data_group(model, group)

    def residual(parameter, group, model, data, **kwargs):
        parameter = ParameterGroup.from_parameter_dict(parameter)
        res = np.concatenate([qr_residual(matrix.T, data[i]) for i, matrix in
                              calculate_group(group, model, parameter)])
        return res

    parameter = parameter.as_parameter_dict(only_fit=True)
    minimizer = Minimizer(
        residual,
        parameter,
        fcn_args=[group, model, data],
        fcn_kws=None,
        iter_cb=None,
        scale_covar=True, nan_policy='omit',
        reduce_fcn=None,
        **{})

    lm_result = minimizer.minimize(method='least_squares', verbose=2, max_nfev=None)
    parameter = ParameterGroup.from_parameter_dict(lm_result.params)

    dataset_results = {}

    for label in model.dataset:
        filled_dataset = model.dataset[label].fill(model, parameter)
        dataset = model.dataset[label].dataset
        data = dataset.get()

        calculated_axis = dataset.get_axis(model.calculated_axis)
        estimated_axis = dataset.get_axis(model.estimated_axis)

        calculated_matrix = \
            [model.calculated_matrix(filled_dataset, model.compartment, index, calculated_axis)
             for index in estimated_axis]

        compartments = calculated_matrix[0][0]

        dim1 = len(calculated_matrix)
        dim2 = len(compartments)

        calculated_matrix = [c[1] for c in calculated_matrix]
        estimated_matrix = np.empty((dim1, dim2), dtype=np.float64)
        for i in range(dim1):
            estimated_matrix[i, :] = \
                   qr_coefficents(calculated_matrix[i].T, data[i, :])[:dim2]

        dim2 = calculated_matrix[0].shape[1]
        result = np.zeros((dim1, dim2), dtype=np.float64)
        for i in range(dim1):
            result[i, :] = np.dot(estimated_matrix[i, :], calculated_matrix[i])
        data = Dataset()
        data.set_axis(model.calculated_axis, calculated_axis)
        data.set_axis(model.estimated_axis, estimated_axis)
        data.set(result)
        dataset_results[label] = DatasetResult(label,
                                               compartments,
                                               calculated_matrix,
                                               estimated_matrix,
                                               data)

    return FitResult(lm_result, dataset_results)
