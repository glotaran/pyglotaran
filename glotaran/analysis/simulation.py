"""Functions for simulating a global analysis model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.analysis.util import calculate_matrix

if TYPE_CHECKING:
    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup


def simulate(
    model: Model,
    dataset: str,
    parameters: ParameterGroup,
    axes: dict[str, np.ndarray] = None,
    clp: np.ndarray | xr.DataArray = None,
    noise=False,
    noise_std_dev=1.0,
    noise_seed=None,
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

    if model.global_matrix is None and clp is None:
        raise ValueError(
            "Cannot simulate models without implementation for global matrix and no clp given."
        )

    filled_dataset = model.dataset[dataset].fill(model, parameters)
    filled_dataset.overwrite_global_dimension(model.global_dimension)
    if hasattr(model, "overwrite_index_dependent"):
        filled_dataset.overwrite_index_dependent(model.overwrite_index_dependent())

    model_dimension = filled_dataset.get_model_dimension()
    model_axis = axes[model_dimension]
    global_dimension = filled_dataset.get_global_dimension()
    global_axis = axes[global_dimension]

    result = xr.DataArray(
        data=0.0,
        coords=[
            (model_dimension, model_axis),
            (global_dimension, global_axis),
        ],
    )
    result = result.to_dataset(name="data")

    matrix = (
        [
            calculate_matrix(
                model,
                filled_dataset,
                {global_dimension: index},
                {model_dimension: model_axis, global_dimension: global_axis},
            )
            for index, _ in enumerate(global_axis)
        ]
        if filled_dataset.index_dependent()
        else calculate_matrix(
            model,
            filled_dataset,
            {},
            {model_dimension: model_axis, global_dimension: global_axis},
        )
    )
    if callable(model.constrain_matrix_function):
        matrix = (
            [
                model.constrain_matrix_function(dataset, parameters, clp, mat, global_axis[i])
                for i, (clp, mat) in enumerate(matrix)
            ]
            if filled_dataset.index_dependent()
            else model.constrain_matrix_function(dataset, parameters, matrix[0], matrix[1], None)
        )
    matrix = (
        [
            xr.DataArray(mat, coords=[(model_dimension, model_axis), ("clp_label", clp_label)])
            for clp_label, mat in matrix
        ]
        if filled_dataset.index_dependent()
        else xr.DataArray(
            matrix[1], coords=[(model_dimension, model_axis), ("clp_label", matrix[0])]
        )
    )

    if clp is not None:
        if clp.shape[0] != global_axis.size:
            raise ValueError(
                f"Size of dimension 0 of clp ({clp.shape[0]}) != size of axis"
                f" '{global_dimension}' ({global_axis.size})"
            )
        if isinstance(clp, xr.DataArray):
            if global_dimension not in clp.coords:
                raise ValueError(f"Missing coordinate '{global_dimension}' in clp.")
            if "clp_label" not in clp.coords:
                raise ValueError("Missing coordinate 'clp_label' in clp.")
        else:
            if "clp_label" not in axes:
                raise ValueError("Missing axis 'clp_label'")
            clp = xr.DataArray(
                clp,
                coords=[
                    (global_dimension, global_axis),
                    ("clp_label", axes["clp_label"]),
                ],
            )
    else:
        clp_labels, clp = model.global_matrix(filled_dataset, global_axis)
        clp = xr.DataArray(
            clp, coords=[(global_dimension, global_axis), ("clp_label", clp_labels)]
        )
    for i in range(global_axis.size):
        index_matrix = matrix[i] if filled_dataset.index_dependent() else matrix
        result.data[:, i] = np.dot(
            index_matrix, clp[i].sel(clp_label=index_matrix.coords["clp_label"])
        )

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result["data"] = (
            (model_dimension, global_dimension),
            np.random.normal(result.data, noise_std_dev),
        )

    return result
