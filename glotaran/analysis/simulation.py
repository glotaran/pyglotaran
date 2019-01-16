"""This package contains functions for simulationg a model."""
from typing import Dict
import numpy as np
import xarray as xr

from glotaran.model.parameter_group import ParameterGroup


def simulate(model,  # temp doc fix : "glotaran.model.Model",
             parameter: ParameterGroup,
             dataset: str,
             axis: Dict[str, np.ndarray],
             noise=False,
             noise_std_dev=1.0,
             noise_seed=None,
             ):
    """Simulates the model.

    Parameters
    ----------
    model: The model to simulate
    parameter : glotaran.model.ParameterGroup
        The parameters for the simulation.
    dataset : str
        Label of the dataset to simulate
    axis : Dict[str, np.ndarray]
        A dictory with axis
    noise : bool
        (Default value = False)
    noise_std_dev : float
        (Default value = 1.0)
    noise_seed : default None

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
