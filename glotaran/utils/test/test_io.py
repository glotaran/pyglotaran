"""Tests for glotaran/utils/io.py"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from IPython.core.formatters import format_display_data
from pandas.testing import assert_frame_equal

from glotaran.io import load_dataset
from glotaran.io import save_dataset
from glotaran.optimization.optimize import optimize
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME
from glotaran.testing.simulated_data.shared_decay import SPECTRAL_AXIS
from glotaran.utils.io import DatasetMapping
from glotaran.utils.io import chdir_context
from glotaran.utils.io import create_clp_guide_dataset
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


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


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

    rel_result_no_common = relative_posix_path(
        (tmp_path / f"../{rel_file_path}").resolve().as_posix(), str(tmp_path)
    )

    assert rel_result_no_common == f"../{rel_file_path}"

    result_folder = tmp_path / "results"
    with chdir_context(result_folder):
        original_rel_path = relative_posix_path(rel_file_path, result_folder)
        assert original_rel_path == rel_file_path

        original_rel_path = relative_posix_path(rel_file_path, "not_a_parent")
        assert original_rel_path == rel_file_path


def test_chdir_context(tmp_path: Path):
    """Original Path is restored even after exception is thrown."""
    original_dir = Path(os.curdir).resolve()
    with chdir_context(tmp_path) as curdir:
        assert curdir == tmp_path.resolve()
        assert tmp_path.resolve() == Path(os.curdir).resolve()
        assert Path("test.txt").resolve() == (tmp_path / "test.txt").resolve()

    assert Path(os.curdir).resolve() == original_dir

    with pytest.raises(ValueError):
        with chdir_context(tmp_path):
            raise ValueError("Original path will be restored after I raise.")

    assert Path(os.curdir).resolve() == original_dir


def test_chdir_context_exception(tmp_path: Path):
    """Raise error if ``folder_path`` is an existing file instead of a folder."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    with pytest.raises(ValueError) as excinfo:
        with chdir_context(file_path):
            pass

    assert (
        str(excinfo.value)
        == "Value of 'folder_path' needs to be a folder but was an existing file."
    )


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


def test_create_clp_guide_dataset(dummy_result: Result):
    """Check that clp guide has correct dimensions and dimension values."""
    clp_guide = create_clp_guide_dataset(dummy_result, "species_1", "dataset_1")

    assert clp_guide.data.shape == (1, dummy_result.data["dataset_1"].spectral.size)
    assert np.allclose(clp_guide.coords["time"].item(), -1)
    assert np.allclose(clp_guide.coords["spectral"].values, SPECTRAL_AXIS)

    clp_guide = create_clp_guide_dataset(dummy_result.data["dataset_1"], "species_1")

    assert clp_guide.data.shape == (1, dummy_result.data["dataset_1"].spectral.size)
    assert np.allclose(clp_guide.coords["time"].item(), -1)
    assert np.allclose(clp_guide.coords["spectral"].values, SPECTRAL_AXIS)


def test_create_clp_guide_dataset_errors(dummy_result: Result):
    """Errors thrown when dataset or clp_label are not in result."""
    with pytest.raises(ValueError) as exc_info:
        create_clp_guide_dataset(dummy_result, "species_1", "not-a-dataset")

    assert (
        str(exc_info.value)
        == "Unknown dataset 'not-a-dataset'. Known datasets are:\n ['dataset_1']"
    )

    with pytest.raises(ValueError) as exc_info:
        create_clp_guide_dataset(dummy_result, "not-a-species", "dataset_1")

    assert (
        str(exc_info.value) == "Unknown clp_label 'not-a-species'. Known clp_labels are:\n "
        "['species_1', 'species_2', 'species_3']"
    )

    dummy_dataset = dummy_result.data["dataset_1"].copy()
    del dummy_dataset.attrs["model_dimension"]

    with pytest.raises(ValueError) as exc_info:
        create_clp_guide_dataset(dummy_dataset, "species_1")

    assert (
        str(exc_info.value) == "Result dataset is missing attribute 'model_dimension', "
        "which means that it was created with pyglotaran<0.6.0."
        "Please recreate the result with the latest version of pyglotaran."
    )


def test_extract_sas_ascii_round_trip(dummy_result: Result, tmp_path: Path):
    """Save to and then load from ascii results in the same data (spectrum)."""
    tmp_file = tmp_path / "sas.ascii"

    sas = create_clp_guide_dataset(dummy_result, "species_1", "dataset_1")
    with pytest.warns(UserWarning) as rec_warn:
        save_dataset(sas, tmp_file)

        assert len(rec_warn) == 1
        assert Path(rec_warn[0].filename).samefile(__file__)
        assert rec_warn[0].message.args[0] == (
            "Saving the 'data' attribute of 'dataset' as a fallback."
            "Result saving for ascii format only supports xarray.DataArray format, "
            "please pass a xarray.DataArray instead of a xarray.Dataset (e.g. dataset.data)."
        )

    loaded_sas = load_dataset(tmp_file, prepare=False)

    for dim in sas.dims:
        assert all(sas.coords[dim] == loaded_sas.coords[dim]), f"Coordinate {dim} mismatch"
    assert np.allclose(sas.data.values, loaded_sas.data.values)
