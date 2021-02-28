import typing

import numpy as np
import xarray as xr


def prepare_time_trace_dataset(
    dataset: typing.Union[xr.DataArray, xr.Dataset],
    weight: np.ndarray = None,
    irf: typing.Union[np.ndarray, xr.DataArray] = None,
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

    if "data_singular_values" not in dataset:
        l, s, r = np.linalg.svd(dataset.data, full_matrices=False)
        dataset["data_left_singular_vectors"] = (("time", "left_singular_value_index"), l)
        dataset["data_singular_values"] = (("singular_value_index"), s)
        dataset["data_right_singular_vectors"] = (("right_singular_value_index", "spectral"), r)

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
