"""Tests for the glotaran.io.prepare_dataset module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.io.prepare_dataset import add_svd_to_dataset


@pytest.fixture
def data_array() -> xr.DataArray:
    """Return a small 2D data array with explicit time and spectral coordinates."""

    return xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        dims=("time", "spectral"),
        coords={"time": [0.0, 1.0, 2.0], "spectral": [700.0, 710.0]},
    )


def test_add_svd_to_dataset_adds_expected_variables(data_array: xr.DataArray):
    """Add SVD outputs for the default dataset variable and dimensions."""

    dataset = data_array.to_dataset(name="data")

    add_svd_to_dataset(dataset)

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        data_array.data, full_matrices=False
    )

    assert dataset.data_left_singular_vectors.dims == ("time", "left_singular_value_index")
    assert dataset.data_singular_values.dims == ("singular_value_index",)
    assert dataset.data_right_singular_vectors.dims == (
        "spectral",
        "right_singular_value_index",
    )
    np.testing.assert_allclose(dataset.data_left_singular_vectors.to_numpy(), left_vectors)
    np.testing.assert_allclose(dataset.data_singular_values.to_numpy(), singular_values)
    np.testing.assert_allclose(dataset.data_right_singular_vectors.to_numpy(), right_vectors.T)


def test_add_svd_to_dataset_works_with_transposed_data(data_array: xr.DataArray):
    """Exercise the transposed-data path using the dataset's default variable."""

    dataset = data_array.T.to_dataset(name="data")

    add_svd_to_dataset(dataset)

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        data_array.data, full_matrices=False
    )

    assert dataset.data_left_singular_vectors.dims == ("time", "left_singular_value_index")
    assert dataset.data_singular_values.dims == ("singular_value_index",)
    assert dataset.data_right_singular_vectors.dims == (
        "spectral",
        "right_singular_value_index",
    )
    np.testing.assert_allclose(dataset.data_left_singular_vectors.to_numpy(), left_vectors)
    np.testing.assert_allclose(dataset.data_singular_values.to_numpy(), singular_values)
    np.testing.assert_allclose(dataset.data_right_singular_vectors.to_numpy(), right_vectors.T)


def test_add_svd_to_dataset_uses_custom_name_dims_and_data_array(data_array: xr.DataArray):
    """Use a provided data array and custom output names for the SVD variables."""

    dataset = xr.Dataset()

    add_svd_to_dataset(
        dataset,
        name="fitted_data",
        lsv_dim="model_time",
        rsv_dim="wavelength",
        data_array=data_array.rename(time="model_time", spectral="wavelength"),
    )

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        data_array.data, full_matrices=False
    )

    assert dataset.fitted_data_left_singular_vectors.dims == (
        "model_time",
        "left_singular_value_index",
    )
    assert dataset.fitted_data_singular_values.dims == ("singular_value_index",)
    assert dataset.fitted_data_right_singular_vectors.dims == (
        "wavelength",
        "right_singular_value_index",
    )
    np.testing.assert_allclose(dataset.fitted_data_left_singular_vectors.to_numpy(), left_vectors)
    np.testing.assert_allclose(dataset.fitted_data_singular_values.to_numpy(), singular_values)
    np.testing.assert_allclose(
        dataset.fitted_data_right_singular_vectors.to_numpy(), right_vectors.T
    )


def test_add_svd_to_dataset_skips_recomputing_existing_svd(
    data_array: xr.DataArray, monkeypatch: pytest.MonkeyPatch
):
    """Skip the SVD calculation when singular values already exist on the dataset."""

    dataset = data_array.to_dataset(name="data")
    dataset["data_singular_values"] = (("singular_value_index",), np.array([42.0, 24.0]))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("np.linalg.svd should not be called when SVD data already exists")

    monkeypatch.setattr(np.linalg, "svd", fail_if_called)

    add_svd_to_dataset(dataset)

    np.testing.assert_allclose(dataset.data_singular_values.to_numpy(), np.array([42.0, 24.0]))
    assert "data_left_singular_vectors" not in dataset
    assert "data_right_singular_vectors" not in dataset
