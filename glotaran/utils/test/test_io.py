"""Tests for glotaran/utils/io.py"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from IPython.core.formatters import format_display_data
from pandas.testing import assert_frame_equal

from glotaran.analysis.optimize import optimize
from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.plugin_system.data_io_registration import save_dataset
from glotaran.project.scheme import Scheme
from glotaran.testing.simulated_data.parallel_spectral_decay import SCHEME as par_scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME as seq_scheme
from glotaran.utils.io import DatasetMapping
from glotaran.utils.io import extract_sas
from glotaran.utils.io import load_datasets
from glotaran.utils.io import relative_posix_path
from glotaran.utils.io import safe_dataframe_fillna
from glotaran.utils.io import safe_dataframe_replace


@pytest.fixture
def ds_mapping() -> DatasetMapping:
    """Dummy mapping for testing."""
    ds_mapping = DatasetMapping()

    ds_mapping["ds1"] = xr.DataArray([1, 2]).to_dataset(name="data")
    ds_mapping["ds2"] = xr.DataArray([3, 4]).to_dataset(name="data")
    return ds_mapping


@pytest.fixture
def dummy_datasets(tmp_path: Path) -> tuple[Path, xr.Dataset, xr.Dataset]:
    """Dummy files for testing."""
    ds1 = xr.DataArray([1, 2]).to_dataset(name="data")
    ds2 = xr.DataArray([3, 4]).to_dataset(name="data")
    save_dataset(ds1, tmp_path / "ds1_file.nc")
    save_dataset(ds2, tmp_path / "ds2_file.nc")
    return tmp_path, ds1, ds2


def test_dataset_mapping(ds_mapping: DatasetMapping):
    """Basic mapping functionality of ``DatasetMapping``."""

    assert "ds1" in ds_mapping
    assert "ds2" in ds_mapping
    assert len(ds_mapping) == 2

    assert repr(ds_mapping) == "{'ds1': <xarray.Dataset>, 'ds2': <xarray.Dataset>}"

    for ds_name, expected_ds_name in zip(ds_mapping, ["ds1", "ds2"]):
        assert ds_name == expected_ds_name

    del ds_mapping["ds1"]

    assert "ds1" not in ds_mapping
    assert "ds2" in ds_mapping
    assert len(ds_mapping) == 1

    assert repr(ds_mapping) == "{'ds2': <xarray.Dataset>}"


def test_dataset_mapping_ipython_render(ds_mapping: DatasetMapping):
    """Renders as html in an ipython context."""

    rendered_result = format_display_data(ds_mapping)[0]

    assert "text/html" in rendered_result
    assert html.unescape(rendered_result["text/html"]).startswith(
        "<pre>{'ds1': <xarray.Dataset>, 'ds2': <xarray.Dataset>}</pre>"
        "\n<details><summary>ds1</summary>"
    )
    assert rendered_result["text/plain"] == repr(ds_mapping)


def test_load_datasets_single_dataset(dummy_datasets: tuple[Path, xr.Dataset, xr.Dataset]):
    """Functionality of ``load_datasets`` with a single dataset of all supported types."""
    tmp_path, ds1, _ = dummy_datasets
    expected_source_path = (tmp_path / "ds1_file.nc").as_posix()

    str_result = load_datasets((tmp_path / "ds1_file.nc").as_posix())

    assert "ds1_file" in str_result
    assert np.all(str_result["ds1_file"].data == ds1.data)
    assert str_result["ds1_file"].source_path == expected_source_path
    assert str_result.source_path["ds1_file"] == expected_source_path

    path_result = load_datasets(tmp_path / "ds1_file.nc")

    assert "ds1_file" in path_result
    assert np.all(path_result["ds1_file"].data == ds1.data)
    assert path_result["ds1_file"].source_path == expected_source_path
    assert path_result.source_path["ds1_file"] == expected_source_path

    dataset_result = load_datasets(ds1)

    assert "ds1_file" in dataset_result
    assert np.all(dataset_result["ds1_file"].data == ds1.data)
    assert dataset_result["ds1_file"].source_path == expected_source_path
    assert dataset_result.source_path["ds1_file"] == expected_source_path

    dataarray_result = load_datasets(xr.DataArray([1, 2]))

    assert "dataset_1" in dataarray_result
    assert np.all(dataarray_result["dataset_1"].data == ds1.data)
    assert dataarray_result["dataset_1"].source_path == "dataset_1.nc"
    assert dataarray_result.source_path["dataset_1"] == "dataset_1.nc"

    pure_dataset_result = load_datasets(xr.DataArray([1, 2]).to_dataset(name="data"))

    assert "dataset_1" in pure_dataset_result
    assert np.all(pure_dataset_result["dataset_1"].data == ds1.data)
    assert pure_dataset_result["dataset_1"].source_path == "dataset_1.nc"
    assert pure_dataset_result.source_path["dataset_1"] == "dataset_1.nc"


def test_load_datasets_sequence(dummy_datasets: tuple[Path, xr.Dataset, xr.Dataset]):
    """Functionality of ``load_datasets`` with a sequence."""
    tmp_path, ds1, ds2 = dummy_datasets

    result = load_datasets([tmp_path / "ds1_file.nc", tmp_path / "ds2_file.nc"])

    assert "ds1_file" in result
    assert np.all(result["ds1_file"].data == ds1.data)
    assert result["ds1_file"].source_path == (tmp_path / "ds1_file.nc").as_posix()
    assert result.source_path["ds1_file"] == (tmp_path / "ds1_file.nc").as_posix()

    assert "ds2_file" in result
    assert np.all(result["ds2_file"].data == ds2.data)
    assert result["ds2_file"].source_path == (tmp_path / "ds2_file.nc").as_posix()
    assert result.source_path["ds2_file"] == (tmp_path / "ds2_file.nc").as_posix()


def test_load_datasets_mapping(dummy_datasets: tuple[Path, xr.Dataset, xr.Dataset]):
    """Functionality of ``load_datasets`` with a mapping."""
    tmp_path, ds1, ds2 = dummy_datasets

    result = load_datasets({"ds1": tmp_path / "ds1_file.nc", "ds2": tmp_path / "ds2_file.nc"})

    assert "ds1" in result
    assert np.all(result["ds1"].data == ds1.data)
    assert result["ds1"].source_path == (tmp_path / "ds1_file.nc").as_posix()
    assert result.source_path["ds1"] == (tmp_path / "ds1_file.nc").as_posix()

    assert "ds2" in result
    assert np.all(result["ds2"].data == ds2.data)
    assert result["ds2"].source_path == (tmp_path / "ds2_file.nc").as_posix()
    assert result.source_path["ds2"] == (tmp_path / "ds2_file.nc").as_posix()


def test_load_datasets_wrong_type():
    """Raise TypeError for not supported type"""
    with pytest.raises(
        TypeError,
        match=(
            r"Type 'int' for 'dataset_mappable' of value "
            r"'1' is not supported\."
            r"\nSupported types are:\n"
        ),
    ):
        load_datasets(1)


@pytest.mark.parametrize("rel_file_path", ("file.txt", "folder/file.txt"))
def test_relative_posix_path(tmp_path: Path, rel_file_path: str):
    """All possible permutation for the input values."""
    full_path = tmp_path / rel_file_path

    result_str = relative_posix_path(str(full_path))

    assert result_str == full_path.as_posix()

    result_path = relative_posix_path(full_path)

    assert result_path == full_path.as_posix()

    rel_result_str = relative_posix_path(str(full_path), tmp_path)

    assert rel_result_str == rel_file_path

    rel_result_path = relative_posix_path(full_path, str(tmp_path))

    assert rel_result_path == rel_file_path

    rel_result_no_coomon = relative_posix_path(
        (tmp_path / f"../{rel_file_path}").resolve().as_posix(), str(tmp_path)
    )

    assert rel_result_no_coomon == f"../{rel_file_path}"


@pytest.mark.skipif(not sys.platform.startswith("win32"), reason="Only needed for Windows")
def test_relative_posix_path_windows_diff_drives():
    """os.path.relpath doesn't cause crash when files are on different drives."""

    source_path = "D:\\data\\data_file.txt"
    result = relative_posix_path(source_path, "C:\\result_path")

    assert result == Path(source_path).as_posix()


def test_safe_dataframe_fillna():
    """Only values in selected columns are filled"""
    df = pd.DataFrame(
        {
            "stays_same": [np.nan, -4, 2],
            "minimum": [np.nan, -4, 2],
            "maximum": [np.nan, -4, 2],
        }
    )

    df2 = pd.DataFrame(
        {
            "stays_same": [np.nan, -4, 2],
            "minimum": [-np.inf, -4, 2],
            "maximum": [np.inf, -4, 2],
        }
    )

    safe_dataframe_fillna(df, "minimum", -np.inf)
    safe_dataframe_fillna(df, "maximum", np.inf)
    safe_dataframe_fillna(df, "not_a_column", np.inf)

    assert_frame_equal(df, df2)


def test_safe_dataframe_replace():
    """Only values in selected columns are replaced"""

    df = pd.DataFrame(
        {
            "stays_same": [np.nan, -4, 2],
            "minimum": [-np.inf, -4, 2],
            "maximum": [np.inf, -4, 2],
            "list_test": [np.inf, -4, 2],
            "tuple_test": [np.inf, -4, 2],
        }
    )

    df2 = pd.DataFrame(
        {
            "stays_same": [np.nan, -4, 2],
            "minimum": ["", -4, 2],
            "maximum": ["", -4, 2],
            "list_test": [1.0, 1, 2],
            "tuple_test": [3.0, 3, 2],
        }
    )

    safe_dataframe_replace(df, "minimum", -np.inf, "")
    safe_dataframe_replace(df, "maximum", (np.inf, np.nan), "")
    safe_dataframe_replace(df, "list_test", (np.inf, -4), 1)
    safe_dataframe_replace(df, "tuple_test", (np.inf, -4), 3)
    safe_dataframe_replace(df, "not_a_column", np.inf, 2)

    assert_frame_equal(df, df2)


@pytest.mark.parametrize("scheme", (seq_scheme, par_scheme))
def test_extract_sas(scheme: Scheme):
    """Same spectral dimension and data values as direct selected data."""
    result = optimize(scheme)
    result_dataset = result.data["dataset_1"]

    sas = extract_sas(result, "dataset_1", "species_1")

    assert sas.coords["time"] == [0]
    assert np.all(sas.coords["spectral"] == result_dataset.coords["spectral"])
    assert np.all(
        sas.values[0] == result_dataset.species_associated_spectra.sel(species="species_1").values
    )

    sas_from_dataset = extract_sas(result_dataset, species="species_1")

    assert np.all(
        sas_from_dataset.values[0]
        == result_dataset.species_associated_spectra.sel(species="species_1").values
    )


def test_extract_sas_ascii_round_trip(tmp_path: Path):
    """Save and load from ascii give same result."""
    result = optimize(seq_scheme)
    tmp_file = tmp_path / "sas.ascii"

    sas = extract_sas(result, "dataset_1", "species_1")
    save_dataset(sas, tmp_file)
    loaded_sas = load_dataset(tmp_file, prepare=False)
    del sas.attrs["loader"]
    del sas.attrs["source_path"]

    for dim in sas.dims:
        assert all(sas.coords[dim] == loaded_sas.coords[dim]), f"Coordinate {dim} mismatch"
    assert np.allclose(sas.values, loaded_sas.data.values)


def test_extract_sas_exceptions():
    """Raise error with usage help on wrong dataset name or species."""
    result = optimize(seq_scheme)

    with pytest.raises(ValueError) as exec_info:
        extract_sas(result, "not_a_dataset")

    assert str(exec_info.value) == (
        "The result doesn't contain a dataset with name 'not_a_dataset'.\n"
        "Valid values are: ['dataset_1']"
    )

    with pytest.raises(ValueError) as exec_info:
        extract_sas("result")

    assert str(exec_info.value) == (
        "Unsupported result type: 'str'\nSupported types are: ['Result', 'xr.Dataset']"
    )

    with pytest.raises(ValueError) as exec_info:
        extract_sas(result, "dataset_1")

    assert str(exec_info.value) == (
        "The result doesn't contain a species with name None.\n"
        "Valid values are: ['species_1', 'species_2', 'species_3']"
    )

    with pytest.raises(ValueError) as exec_info:
        extract_sas(result, "dataset_1", "s1")

    assert str(exec_info.value) == (
        "The result doesn't contain a species with name 's1'.\n"
        "Valid values are: ['species_1', 'species_2', 'species_3']"
    )

    bad_result = xr.Dataset(
        {"foo": (("species", "spectral"), [[1, 2], [3, 4]])},
        coords={"species": ["species_1", "species_2"], "spectral": [1, 2]},
    )

    with pytest.raises(ValueError) as exec_info:
        extract_sas(bad_result, species="species_1")

    assert str(exec_info.value) == (
        "The result does not have a 'species_associated_spectra' data variable.\n"
        "Contained data variables are: ['foo']"
    )
