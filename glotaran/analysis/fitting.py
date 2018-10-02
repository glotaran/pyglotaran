"""This package contains the fit functions."""

from typing import Dict, List
import numpy as np
from lmfit import Minimizer

from ..datasets.dataset import Dataset
from ..model.parameter_group import ParameterGroup

from .grouping import Group, create_group, calculate_group, create_data_group
from .variable_projection import qr_residual, qr_coefficents
from .fitresult import DatasetResult, FitResult


def residual_variable_projection(parameter: ParameterGroup,
                                 group: Group,
                                 model: 'glotaran.model.BaseModel',
                                 data: Dict[str, Dataset],
                                 data_group: List[np.ndarray],
                                 **kwargs) -> np.ndarray:
    """residul_variable_projection returns the variable projection residual.

    Parameters
    ----------
    parameter : ParameterGroup
    group: Dict[any, Tuple[any, DatasetDescriptor]]
    model : glotaran.model.BaseModel
    data : Dict[str, Dataset]
        A dictionary of dataset labels and Datasets.
    data_group : List[np.ndarray]
    **kwargs

    Returns
    -------
    residual: np.ndarray
    """
    res = np.concatenate([qr_residual(matrix.T, data_group[i]) for i, matrix in
                          calculate_group(group, model, parameter, data)])
    return res


def fit(model,  # temp doc fix : 'glotaran.model.BaseModel',
        parameter: ParameterGroup,
        data: Dict[str, Dataset],
        verbose: int = 2,
        max_fnev: int = None,) -> FitResult:
    """fit performs a fit of the model.

    Parameters
    ----------
    model : glotaran.model.BaseModel
        The Model to fit.
    parameter : ParameterGroup
        The initial fit parameter.
    data : Dict[str, Dataset],
        A dictionary of dataset labels and Datasets.
    verbose : int
        (optional default=2)
        Set 0 for no log output, 1 for only result and 2 for full verbosity
    max_fnev : int
        (optional default=None)
        The maximum number of function evaluations, default None.

    Returns
    -------
    result: FitResult
        The result of the fit.
    """

    group = create_group(model, data)
    data_group = create_data_group(model, group, data)

    def residual_proxy(parameter, group, model, data, data_group, **kwargs):
        parameter = ParameterGroup.from_parameter_dict(parameter)
        return residual_variable_projection(parameter, group, model, data, data_group, **kwargs)

    parameter = parameter.as_parameter_dict(only_fit=True)
    minimizer = Minimizer(
        residual_proxy,
        parameter,
        fcn_args=[group, model, data, data_group],
        fcn_kws=None,
        iter_cb=None,
        scale_covar=True,
        nan_policy='omit',
        reduce_fcn=None,
        **{})

    lm_result = minimizer.minimize(method='least_squares',
                                   verbose=verbose,
                                   max_nfev=max_fnev)
    parameter = ParameterGroup.from_parameter_dict(lm_result.params)

    dataset_results = {}

    for label in model.dataset:
        filled_dataset = model.dataset[label].fill(model, parameter)
        dataset = data[label]

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
                   qr_coefficents(calculated_matrix[i].T, dataset.data()[i, :])[:dim2]

        dim2 = calculated_matrix[0].shape[1]
        result = np.zeros((dim1, dim2), dtype=np.float64)
        for i in range(dim1):
            result[i, :] = np.dot(estimated_matrix[i, :], calculated_matrix[i])
        dataset = Dataset()
        dataset.set_axis(model.calculated_axis, calculated_axis)
        dataset.set_axis(model.estimated_axis, estimated_axis)
        dataset.set_data(result)
        dataset_results[label] = DatasetResult(label,
                                               compartments,
                                               calculated_matrix,
                                               estimated_matrix,
                                               dataset)

    return FitResult(lm_result, dataset_results)
