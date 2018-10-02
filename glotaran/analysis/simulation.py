"""This package contains functions for simulationg a model."""
from typing import Dict
import numpy as np

from glotaran.model.dataset import Dataset
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
    parameter : ParameterGroup
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
        raise Exception("Cannot simulate models without function estimated matrix.")

    filled_dataset = model.dataset[dataset].fill(model, parameter)

    calculated_axis = axis[model.calculated_axis]

    estimated_axis = axis[model.estimated_axis]

    calculated_matrix = [model.calculated_matrix(filled_dataset,
                                                 model.compartment,
                                                 index,
                                                 calculated_axis)
                         for index in estimated_axis]

    compartments = calculated_matrix[0][0]
    estimated_matrix = model.estimated_matrix(filled_dataset, compartments, estimated_axis)

    dim1 = len(calculated_matrix)
    dim2 = calculated_matrix[0][1].shape[1]
    result = np.empty((dim1, dim2), dtype=np.float64)
    for i in range(dim1):
        result[i, :] = np.dot(estimated_matrix[i, :], calculated_matrix[i][1])

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result = np.random.normal(result, noise_std_dev)
    data = Dataset()
    data.set_axis(model.calculated_axis, calculated_axis)
    data.set_axis(model.estimated_axis, estimated_axis)
    data.set_data(result)

    return data
