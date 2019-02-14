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

    if model.global_matrix is None and clp is None:
        raise Exception("Cannot simulate models without implementation for global matrix"
                        " and no clp given.")

    filled_dataset = model.dataset[dataset].fill(model, parameter)

    matrix_dimension = axis[model.matrix_dimension]

    global_dimension = axis[model.global_dimension]

    matrix = [model.matrix(filled_dataset, index, matrix_dimension) for index in global_dimension]
    if callable(model._constrain_matrix_function):
        matrix = [model._constrain_matrix_function(parameter, clp, mat, global_dimension[i])
                  for i, (clp, mat) in enumerate(matrix)]
    matrix = [xr.DataArray(mat, coords=[(model.matrix_dimension, matrix_dimension),
                                        ('clp_label', clp_label)])
              for clp_label, mat in matrix]

    if clp:
        if clp.shape[0] != global_dimension.size:
            raise ValueError(f"Size of dimension 0 of clp ({clp.shape[0]}) != size of axis"
                             f" '{model.global_dimension}' ({global_dimension.size})")
        if isinstance(clp, xr.DataArray):
            if model.global_dimension not in clp.coords:
                raise ValueError(f"Missing coordinate '{model.global_dimension}' in clp.")
            if 'clp_label' not in clp.coords:
                raise ValueError(f"Missing coordinate 'clp_label' in clp.")
        else:
            if 'clp_label' not in axis:
                raise ValueError("Missing axis 'clp_label'")
            clp = xr.DataArray(clp, coords=[(model.global_dimension, global_dimension),
                                            ('clp_label', axis['clp_label'])])
    else:
        clp_labels, clp = model.global_matrix(filled_dataset, global_dimension)
        clp = xr.DataArray(clp, coords=[(model.global_dimension, global_dimension),
                                        ('clp_label', clp_labels)])

    dim1 = matrix_dimension.size
    dim2 = global_dimension.size
    result = np.empty((dim1, dim2), dtype=np.float64)
    for i in range(dim2):
        result[:, i] = np.dot(matrix[i], clp[i])

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result = np.random.normal(result, noise_std_dev)
    data = xr.DataArray(result, coords=[
        (model.matrix_dimension, matrix_dimension), (model.global_dimension, global_dimension)
    ])

    return data.to_dataset(name="data")
