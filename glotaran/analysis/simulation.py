"""This package contains functions for simulating a model."""

import typing
import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup


def simulate(model: typing.Type['glotaran.model.Model'],
             parameter: ParameterGroup,
             dataset: str,
             axis: typing.Dict[str, np.ndarray],
             clp: typing.Union[np.ndarray, xr.DataArray] = None,
             noise=False,
             noise_std_dev=1.0,
             noise_seed=None,
             ):
    """Simulates the model.

    Parameters
    ----------
    model :
        The model to simulate
    parameter :
        The parameters for the simulation.
    dataset :
        Label of the dataset to simulate
    axis :
        A dictory with axis
    noise :
        Add noise to the simulation.
    noise_std_dev :
        The standard devition for noise simulation.
    noise_seed :
        The seed for the noise simulation.
    """

    if model.estimated_matrix is None and clp is None:
        raise Exception("Cannot simulate models without implementation for estimated matrix"
                        " and no clp given.")

    filled_dataset = model.dataset[dataset].fill(model, parameter)

    calculated_axis = axis[model.calculated_axis]

    estimated_axis = axis[model.estimated_axis]

    calculated_matrix = [model.calculated_matrix(filled_dataset,
                                                 index,
                                                 calculated_axis)
                         for index in estimated_axis]
    if callable(model._constrain_calculated_matrix_function):
        calculated_matrix = [model._constrain_calculated_matrix_function(parameter,
                                                                         clp, mat,
                                                                         estimated_axis[i])
                             for i, (clp, mat) in enumerate(calculated_matrix)]
    calculated_matrix = [xr.DataArray(mat, coords=[(model.calculated_axis, calculated_axis),
                                                   ('clp_label', clp_label)])
                         for clp_label, mat in calculated_matrix]

    if clp:
        if clp.shape[0] != estimated_axis.size:
            raise ValueError(f"Size of dimension 0 of clp ({clp.shape[0]}) != size of axis"
                             f" '{model.estimated_axis}' ({estimated_axis.size})")
        if isinstance(clp, xr.DataArray):
            if model.estimated_axis not in clp.coords:
                raise ValueError(f"Missing coordinate '{model.estimated_axis}' in clp.")
            if 'clp_label' not in clp.coords:
                raise ValueError(f"Missing coordinate 'clp_label' in clp.")
        else:
            if 'clp_label' not in axis:
                raise ValueError("Missing axis 'clp_label'")
            clp = xr.DataArray(clp, coords=[(model.estimated_axis, estimated_axis),
                                            ('clp_label', axis['clp_label'])])
    else:
        clp_labels, clp = model.estimated_matrix(filled_dataset, estimated_axis)
        clp = xr.DataArray(clp, coords=[(model.estimated_axis, estimated_axis),
                                        ('clp_label', clp_labels)])

    dim1 = calculated_axis.size
    dim2 = estimated_axis.size
    result = np.empty((dim1, dim2), dtype=np.float64)
    for i in range(dim2):
        result[:, i] = np.dot(calculated_matrix[i], clp[i])

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result = np.random.normal(result, noise_std_dev)
    data = xr.DataArray(result, coords=[
        (model.calculated_axis, calculated_axis), (model.estimated_axis, estimated_axis)
    ])

    return data.to_dataset(name="data")
