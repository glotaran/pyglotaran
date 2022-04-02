"""Functions for simulating a global analysis model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.analysis.util import calculate_matrix
from glotaran.model import DatasetModel

if TYPE_CHECKING:
    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup


def simulate(
    model: Model,
    dataset: str,
    parameters: ParameterGroup,
    coordinates: dict[str, np.ndarray],
    clp: xr.DataArray | None = None,
    noise: bool = False,
    noise_std_dev: float = 1.0,
    noise_seed: int | None = None,
):
    """Simulates a model.

    Parameters
    ----------
    model :
        The model to simulate.
    parameter :
        The parameters for the simulation.
    dataset :
        Label of the dataset to simulate
    axes :
        A dictionary with axes for simulation.
    clp :
        conditionally linear parameters. Will be used instead of `model.global_matrix` if given.
    noise :
        Add noise to the simulation.
    noise_std_dev :
        The standard deviation for noise simulation.
    noise_seed :
        The seed for the noise simulation.
    """

    dataset_model = model.dataset[dataset].fill(model, parameters)
    dataset_model.set_coordinates(coordinates)

    if dataset_model.has_global_model():
        result = simulate_global_model(
            dataset_model,
            parameters,
            clp,
        )
    elif clp is None:
        raise ValueError(
            f"Cannot simulate dataset {dataset} without global megacomplex " "and no clp provided."
        )
    else:
        result = simulate_clp(
            dataset_model,
            parameters,
            clp,
        )

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result["data"] = (result.data.dims, np.random.normal(result.data, noise_std_dev))

    return result


def simulate_clp(
    dataset_model: DatasetModel,
    parameters: ParameterGroup,
    clp: xr.DataArray,
):

    if "clp_label" not in clp.coords:
        raise ValueError("Missing coordinate 'clp_label' in clp.")
    global_dimension = next(dim for dim in clp.coords if dim != "clp_label")

    global_axis = clp.coords[global_dimension]
    matrices = (
        [
            calculate_matrix(
                dataset_model,
                {global_dimension: index},
            )
            for index, _ in enumerate(global_axis)
        ]
        if dataset_model.is_index_dependent()
        else calculate_matrix(dataset_model, {})
    )

    model_dimension = dataset_model.get_model_dimension()
    model_axis = dataset_model.get_coordinates()[model_dimension]
    result = xr.DataArray(
        data=0.0,
        coords=[
            (model_dimension, model_axis.data),
            (global_dimension, global_axis.data),
        ],
    )
    result = result.to_dataset(name="data")
    for i in range(global_axis.size):
        index_matrix = matrices[i] if dataset_model.is_index_dependent() else matrices
        result.data[:, i] = np.dot(
            index_matrix.matrix,
            clp.isel({global_dimension: i}).sel({"clp_label": index_matrix.clp_labels}),
        )

    return result


def simulate_global_model(
    dataset_model: DatasetModel,
    parameters: ParameterGroup,
    clp: xr.DataArray = None,
):
    """Simulates a global model."""

    # TODO: implement full model clp
    if clp is not None:
        raise NotImplementedError("Simulation of full models with clp is not supported yet.")

    if any(m.index_dependent(dataset_model) for m in dataset_model.global_megacomplex):
        raise ValueError("Index dependent models for global dimension are not supported.")

    global_matrix = calculate_matrix(dataset_model, {}, as_global_model=True)
    global_clp_labels = global_matrix.clp_labels
    global_matrix = xr.DataArray(
        global_matrix.matrix.T,
        coords=[
            ("clp_label", global_clp_labels),
            (dataset_model.get_global_dimension(), dataset_model.get_global_axis()),
        ],
    )

    return simulate_clp(
        dataset_model,
        parameters,
        global_matrix,
    )
