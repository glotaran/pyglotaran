from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.io.prepare_dataset import add_svd_to_dataset


@pytest.mark.parametrize("transposed", [False, True])
def test_add_svd_respects_requested_dimension_order(transposed: bool):
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        coords={"time": [1, 2, 3], "spectral": [10, 20, 30, 40]},
    )
    if transposed:
        data = data.transpose("spectral", "time")
    dataset = data.to_dataset(name="data")

    add_svd_to_dataset(dataset)

    expected_lsv, expected_sv, expected_rsv = np.linalg.svd(
        data.transpose("time", "spectral"), full_matrices=False
    )
    assert np.allclose(dataset.data_left_singular_vectors, expected_lsv)
    assert np.allclose(dataset.data_singular_values, expected_sv)
    assert np.allclose(dataset.data_right_singular_vectors, expected_rsv.T)
    assert dataset.data_left_singular_vectors.dims[0] == "time"
    assert dataset.data_right_singular_vectors.dims[0] == "spectral"


def test_add_svd_supports_custom_dimensions_and_data_array():
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3),
        coords={"column": [1, 2], "row": [3, 4, 5]},
    )
    dataset = xr.Dataset()

    add_svd_to_dataset(
        dataset,
        name="custom",
        lsv_dim="row",
        rsv_dim="column",
        data_array=data,
    )

    assert dataset.custom_left_singular_vectors.dims[0] == "row"
    assert dataset.custom_right_singular_vectors.dims[0] == "column"
    assert dataset.custom_singular_values.size == 2


def test_add_svd_does_not_recompute_existing_decomposition():
    dataset = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        coords={"time": [1, 2, 3], "spectral": [10, 20, 30, 40]},
    ).to_dataset(name="data")
    add_svd_to_dataset(dataset)
    singular_values = dataset.data_singular_values.copy()
    dataset["data"] = xr.zeros_like(dataset.data)

    add_svd_to_dataset(dataset)

    assert dataset.data_singular_values.identical(singular_values)
