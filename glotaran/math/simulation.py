from typing import Dict
import numpy as np

from glotaran.model.dataset import Dataset


def simulate(model: "glotaran.model.Model",
             parameter: "glotaran.model.ParameterGroup",
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

    axis : dict(str, np.ndarray)
        A dictory with axis
    noise :
        (Default value = False)
    noise_std_dev :
        (Default value = 1.0)

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

    result = np.asarray([np.dot(estimated_matrix[i, :], c[1])
                         for i, c in enumerate(calculated_matrix)])

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result = np.random.normal(result, noise_std_dev)
    data = Dataset()
    data.set_axis(model.calculated_axis, calculated_axis)
    data.set_axis(model.estimated_axis, estimated_axis)
    data.set(result)

    return data
