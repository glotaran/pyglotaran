from __future__ import annotations

import numpy as np
import xarray as xr


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


def add_svd_to_dataset(dataset: xr.Dataset):
    """Add the SVD of a dataset.data inplace as Data variables to the dataset.

    The SVD is only computed if it doesn't already exist on the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset the SVD values should be added to.
    """

    if "data_singular_values" not in dataset:
        l, s, r = np.linalg.svd(dataset.data, full_matrices=False)
        dataset["data_left_singular_vectors"] = (("time", "left_singular_value_index"), l)
        dataset["data_singular_values"] = (("singular_value_index"), s)
        dataset["data_right_singular_vectors"] = (("right_singular_value_index", "spectral"), r)
