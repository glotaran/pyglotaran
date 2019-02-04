"""This package contains functions for simulating a model."""
import typing
import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup


def simulate(model: typing.Type['glotaran.model.Model'],
             parameter: ParameterGroup,
             dataset: str,
             axis: typing.Dict[str, np.ndarray],
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

    if model.estimated_matrix is None:
        raise Exception("Cannot simulate models without implementation for estimated matrix.")

    filled_dataset = model.dataset[dataset].fill(model, parameter)

    calculated_axis = axis[model.calculated_axis]

    estimated_axis = axis[model.estimated_axis]

    calculated_matrix = [model.calculated_matrix(filled_dataset,
                                                 index,
                                                 calculated_axis)
                         for index in estimated_axis]

    estimated_matrix = model.estimated_matrix(filled_dataset, estimated_axis)

    dim1 = calculated_matrix[0][1].shape[0]
    dim2 = len(calculated_matrix)
    result = np.empty((dim1, dim2), dtype=np.float64)
    for i in range(dim2):
        result[:, i] = np.dot(calculated_matrix[i][1], estimated_matrix[:, i])

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result = np.random.normal(result, noise_std_dev)
    data = xr.DataArray(result, coords=[
        (model.calculated_axis, calculated_axis), (model.estimated_axis, estimated_axis)
    ])

    return data.to_dataset(name="data")
