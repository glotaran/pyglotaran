"""Functions for optimization."""

import typing
import lmfit
import dask
import numpy as np
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
    result :
        The global analysis result.
    verbose :
        If `True` feedback is printed at every iteration.
    max_nfev :
        Maximum number of function evaluations. `None` for unlimited.
    """

    data_groups = {i: dask.delayed(d) for i, d in result.data_groups.items()}
    datasets = dask.delayed(result.data)

    parameter = result.initial_parameter.as_parameter_dict()
    minimizer = lmfit.Minimizer(
        calculate_residual,
        parameter,
        fcn_args=[result, datasets, data_groups],
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


@dask.delayed
def _calculate_group_item(item, model, parameter, data):
    return calculate_group_item(item, model, parameter, data)


@dask.delayed
def _is_finite(item_result, parameter):
    clp_labels = item_result[0]
    matrix = item_result[1]
    for i, row in enumerate(matrix.T):
        if not np.isfinite(row).all():
            raise Exception(f"Matrix is not finite at clp {clp_labels[i]}"
                            f"\n\nCurrent Parameter:\n\n{parameter}")
    return item_result


@dask.delayed
def _residual(item_result, data, nnls):
    matrix = item_result[1]
    residual_func = residual_nnls if nnls else residual_variable_projection
    return residual_func(matrix, data)


@dask.delayed
def _concat(penalty):
    return np.concatenate([p[1] for p in penalty])


def calculate_residual(parameter: typing.Union[ParameterGroup, lmfit.Parameters],
                       result: 'glotaran.analysis.Result', axis, data_groups) -> np.ndarray:
    """Calculates the residual and fills the global analysis result with data.

    Parameters
    ----------
    parameter :
        The parameter for optimization.
    result :
        The global analysis result.
    """

    if not isinstance(parameter, ParameterGroup):
        parameter = ParameterGroup.from_parameter_dict(parameter)

    penalty = []

    if not result.model.index_depended_matrix:
        item = list(result.groups.values())[0]
        item_result = _calculate_group_item(item, result.model, parameter, axis)
        item_result = _is_finite(item_result, parameter)

    for index, item in result.groups.items():
        if result.model.index_depended_matrix:
            item_result = _calculate_group_item(item, result.model, parameter, axis)
            item_result = _is_finite(item_result, parameter)

        residual_result = _residual(item_result, data_groups[index], result.nnls)

        #  result.global_clp[index] = xr.DataArray(clp, coords=[('clp_label', clp_labels)])

        #  start = 0
        #  for i, dataset in item:
        #      dataset = result._data[dataset.label]
        #      if 'residual' not in dataset:
        #          dataset['residual'] = dataset.data.copy()
        #      end = dataset.coords[result.model.matrix_dimension].size + start
        #      dataset.residual.loc[{result.model.global_dimension: i}] = residual[start:end]
        #      start = end
        #
        #      if 'clp' not in dataset:
        #          dim1 = dataset.coords[result.model.global_dimension].size
        #          dim2 = dataset.coords['clp_label'].size
        #          dataset['clp'] = (
        #              (result.model.global_dimension, 'clp_label'),
        #              np.zeros((dim1, dim2), dtype=np.float64)
        #          )
        #      dataset.clp.loc[{result.model.global_dimension: i}] = \
        #          np.array([clp[clp_labels.index(i)] if i in clp_labels else None
        #                    for i in dataset.coords['clp_label'].values])

        #  if callable(result.model._additional_penalty_function):
        #      additionals = result.model._additional_penalty_function(
        #          parameter, clp_labels, clp, matrix, parameter)
        #      residual = np.concatenate((residual, additionals))

        penalty.append(residual_result)
    return _concat(penalty).compute()
