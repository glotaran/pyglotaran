from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import Hashable


def prepare_time_trace_dataset(
    dataset: xr.DataArray | xr.Dataset,
    weight: np.ndarray = None,
    irf: np.ndarray | xr.DataArray = None,
) -> xr.Dataset:
    """Prepares a time trace for global analysis.

    Parameters
    ----------
    dataset :
        The dataset.
    weight :
        A weight for the dataset.
    irf :
        An IRF for the dataset.
    """

    if isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset(name="data")

    add_svd_to_dataset(dataset)

    if weight is not None:
        dataset["weight"] = (dataset.data.dims, weight)
        dataset["weighted_data"] = (dataset.data.dims, np.multiply(dataset.data, dataset.weight))

    if irf is not None:
        if isinstance(irf, np.ndarray):
            if len(irf.shape) != 1:
                raise Exception("IRF with more than one dimension must be `xarray.DataArray`.")
            dataset["irf"] = (("time",), irf)
        else:
            dataset["irf"] = irf

    return dataset


def add_svd_to_dataset(
    dataset: xr.Dataset,
    name: str = "data",
    lsv_dim: Hashable = "time",
    rsv_dim: Hashable = "spectral",
    data_array: xr.DataArray = None,
):
    """Add the SVD of a dataset inplace as Data variables to the dataset.

    The SVD is only computed if it doesn't already exist on the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset the SVD values should be added to.
    name : str
        Key to access the datarray inside of the dataset, by default "data"
    lsv_dim : Hashable
        Name of the dimension for the left singular value, by default "time"
    rsv_dim : Hashable
        Name of the dimension for the right singular value, by default "spectral"
    data_array : xr.DataArray
        Dataarray to calculate the SVD for, when provided the data extraction
        from the dataset will be skipped, by default None
    """
    if data_array is None:
        data_array = dataset[name] if name != "data" else dataset.data
    if f"{name}_singular_values" not in dataset:
        l, s, r = np.linalg.svd(data_array, full_matrices=False)
        dataset[f"{name}_left_singular_vectors"] = ((lsv_dim, "left_singular_value_index"), l)
        dataset[f"{name}_singular_values"] = (("singular_value_index"), s)
        dataset[f"{name}_right_singular_vectors"] = (("right_singular_value_index", rsv_dim), r)
