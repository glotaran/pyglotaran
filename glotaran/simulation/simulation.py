"""Functions for simulating a dataset using a global optimization model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model import DataModel
from glotaran.model import GlotaranUserError
from glotaran.model import Library
from glotaran.model import get_data_model_dimension
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.parameter import Parameters

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def simulate(
    model: DataModel,
    libray: Library,
    parameters: Parameters,
    coordinates: dict[str, ArrayLike],
    clp: xr.DataArray | None = None,
    noise: bool = False,
    noise_std_dev: float = 1.0,
    noise_seed: int | None = None,
) -> xr.Dataset:
    """Simulate a dataset using a model.

    Parameters
    ----------
    model : Model
        The model containing the dataset model.
    dataset : str
        Label of the dataset to simulate
    parameters : Parameters
        The parameters for the simulation.
    coordinates : dict[str, ArrayLike]
        A dictionary with the coordinates used for simulation (e.g. time, wavelengths, ...).
    clp : xr.DataArray | None
        A matrix with conditionally linear parameters (e.g. spectra, pixel intensity, ...).
        Will be used instead of the dataset's global megacomplexes if not None.
    noise : bool
        Add noise to the simulation.
    noise_std_dev : float
        The standard deviation for noise simulation.
    noise_seed : int | None
        The seed for the noise simulation.

    Returns
    -------
    xr.Dataset
        The simulated dataset.


    Raises
    ------
    ValueError
        Raised if dataset model has no global megacomplex and no clp are provided.
    """
    model = libray.resolve_item(model, parameters)
    model_dimension = get_data_model_dimension(model)
    model_axis = coordinates[model_dimension]
    global_dimension = next(dim for dim in coordinates if dim != model_dimension)
    global_axis = coordinates[global_dimension]

    if clp is None:
        if not model.global_megacomplex:
            raise GlotaranUserError(
                "Cannot simulate dataset without global megacomplexes if no clp are provided."
            )
        global_matrix = OptimizationMatrix.from_data_model(
            model, global_axis, model_axis, None, global_matrix=True
        )
        clp = xr.DataArray(
            global_matrix.array,
            coords=((global_dimension, global_axis), ("clp_label", global_matrix.clp_labels)),
        )

    matrix = OptimizationMatrix.from_data_model(model, global_axis, model_axis, None)
    result = xr.DataArray(
        np.zeros((model_axis.size, global_axis.size)),
        coords=[
            (model_dimension, model_axis),
            (global_dimension, global_axis),
        ],
    )
    for i in range(global_axis.size):
        result[:, i] = np.dot(
            matrix.at_index(i).array,
            clp.isel({global_dimension: i}).sel({"clp_label": matrix.clp_labels}),
        )

    if noise and noise_seed is not None:
        np.random.seed(noise_seed)
        result = xr.DataArray(np.random.normal(result.data, noise_std_dev), coords=result.coords)

    return result.to_dataset(name="data")
