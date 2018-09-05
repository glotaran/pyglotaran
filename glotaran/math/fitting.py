from lmfit import Minimizer

import numpy as np
from glotaran.model import ParameterGroup
from .grouping import create_group, calculate_group, get_data_group
from .qr_decomposition import qr_residual
from .result import Result


def fit(model, parameter):

    group = create_group(model)
    data = get_data_group(model, group)
    def residual(parameter, group, model, data, **kwargs):
        parameter = ParameterGroup.from_parameter_dict(parameter)
        group = calculate_group(group, model, parameter)
        res = [qr_residual(group[i], data[i]) for i in range(len(group))]
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
    return Result(minimizer.minimize())
