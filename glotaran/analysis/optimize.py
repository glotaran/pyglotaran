import lmfit
import numpy as np
import typing
import xarray as xr

import glotaran
from glotaran.parameter import ParameterGroup

from .grouping import calculate_group_item
from .nnls import residual_nnls
from .variable_projection import residual_variable_projection


def optimize(result: 'glotaran.analysis.Result', verbose: bool = True, max_nfev: int = None):
    """Optimizes the parameter.

    Parameters
    ----------
    verbose : bool, optional
        (default = True)
        If `True` feedback is printed at every iteration.
    max_nfev : int, optional
        (default = None)
        Maximum number of function evaluations. `None` for unlimited.

    Returns
    -------
    """
    parameter = result.initial_parameter.as_parameter_dict()
    minimizer = lmfit.Minimizer(
        calculate_residual,
        parameter,
        fcn_args=[result],
        fcn_kws=None,
        iter_cb=None,
        scale_covar=True,
        nan_policy='omit',
        reduce_fcn=None,
        **{})
    verbose = 2 if verbose else 0
    lm_result = minimizer.minimize(method='least_squares',
                                   verbose=verbose,
                                   max_nfev=max_nfev)

    result.finalize(lm_result)


def calculate_residual(parameter: typing.Union[ParameterGroup, lmfit.Parameters],
                       result: 'glotaran.analysis.Result'):

    if not isinstance(parameter, ParameterGroup):
        parameter = ParameterGroup.from_parameter_dict(parameter)

    penalty = []
    for index, item in result.groups.items():
        clp_labels, matrix = calculate_group_item(item, result.model, parameter, result.data)

        for i, row in enumerate(matrix.T):
            if not np.isfinite(matrix).all():
                raise Exception(f"Matrix is not finite at clp {clp_labels[i]}"
                                f"\n\nCurrent Parameter:\n\n{parameter}")

        clp = None
        residual = None
        if result.nnls:
            clp, residual = residual_nnls(
                    matrix,
                    result.data_groups[index]
                )
        else:
            clp, residual = residual_variable_projection(
                    matrix,
                    result.data_groups[index]
                )

        result.global_clp[index] = xr.DataArray(clp, coords=[('clp_label', clp_labels)])

        start = 0
        for i, dataset in item:
            dataset = result._data[dataset.label]
            if 'residual' not in dataset:
                dataset['residual'] = dataset.data.copy()
            end = dataset.coords[result.model.matrix_dimension].size + start
            dataset.residual.loc[{result.model.global_dimension: i}] = residual[start:end]
            start = end

            if 'clp' not in dataset:
                dim1 = dataset.coords[result.model.global_dimension].size
                dim2 = dataset.coords['clp_label'].size
                dataset['clp'] = (
                    (result.model.global_dimension, 'clp_label'),
                    np.zeros((dim1, dim2), dtype=np.float64)
                )
            dataset.clp.loc[{result.model.global_dimension: i}] = \
                np.array([clp[clp_labels.index(i)] if i in clp_labels else None
                          for i in dataset.coords['clp_label'].values])

        if callable(result.model._additional_penalty_function):
            additionals = result.model._additional_penalty_function(
                parameter, clp_labels, clp, matrix, parameter)
            residual = np.concatenate((residual, additionals))

        penalty.append(residual)

    return np.concatenate(penalty)
