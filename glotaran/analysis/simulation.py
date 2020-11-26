"""Functions for simulating a global analysis model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import Dict
    from typing import Union

    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup


def simulate(
    model: Model,
    dataset: str,
    parameter: ParameterGroup,
    axes: Dict[str, np.ndarray] = None,
    clp: Union[np.ndarray, xr.DataArray] = None,
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
        Conditionally linear parameter. Will be used instead of `model.global_matrix` if given.
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

    filled_dataset = model.dataset[dataset].fill(model, parameter)

    model_dimension = axes[model.model_dimension]
    global_dimension = axes[model.global_dimension]

    dim1 = model_dimension.size
    dim2 = global_dimension.size
    result = xr.DataArray(
        np.empty((dim1, dim2), dtype=np.float64),
        coords=[
            (model.model_dimension, model_dimension),
            (model.global_dimension, global_dimension),
        ],
    )
    result = result.to_dataset(name="data")

    matrix = [
        model.matrix(dataset_descriptor=filled_dataset, axis=model_dimension, index=index)
        for index in global_dimension
    ]
    if callable(model.constrain_matrix_function):
        matrix = [
            model.constrain_matrix_function(parameter, clp, mat, global_dimension[i])
            for i, (clp, mat) in enumerate(matrix)
        ]
    matrix = [
        xr.DataArray(
            mat, coords=[(model.model_dimension, model_dimension), ("clp_label", clp_label)]
        )
        for clp_label, mat in matrix
    ]

    if clp is not None:
        if clp.shape[0] != global_dimension.size:
            raise ValueError(
                f"Size of dimension 0 of clp ({clp.shape[0]}) != size of axis"
                f" '{model.global_dimension}' ({global_dimension.size})"
            )
        if isinstance(clp, xr.DataArray):
            if model.global_dimension not in clp.coords:
                raise ValueError(f"Missing coordinate '{model.global_dimension}' in clp.")
            if "clp_label" not in clp.coords:
                raise ValueError("Missing coordinate 'clp_label' in clp.")
        else:
            if "clp_label" not in axes:
                raise ValueError("Missing axis 'clp_label'")
            clp = xr.DataArray(
                clp,
                coords=[
                    (model.global_dimension, global_dimension),
                    ("clp_label", axes["clp_label"]),
                ],
            )
    else:
        clp_labels, clp = model.global_matrix(filled_dataset, global_dimension)
        clp = xr.DataArray(
            clp, coords=[(model.global_dimension, global_dimension), ("clp_label", clp_labels)]
        )
    for i in range(dim2):
        result.data[:, i] = np.dot(matrix[i], clp[i].sel(clp_label=matrix[i].coords["clp_label"]))

    if noise:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        result["data"] = (
            (model.model_dimension, model.global_dimension),
            np.random.normal(result.data, noise_std_dev),
        )

    return result
