"""Tests for glotaran/utils/io.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from glotaran.io import save_dataset
from glotaran.utils.io import DatasetMapping
from glotaran.utils.io import load_datasets
from glotaran.utils.io import relative_posix_path


@pytest.fixture
def dummy_datasets(tmp_path: Path) -> tuple[Path, xr.Dataset, xr.Dataset]:
    ds1 = xr.DataArray([1, 2]).to_dataset(name="data")
    ds2 = xr.DataArray([3, 4]).to_dataset(name="data")
    save_dataset(ds1, tmp_path / "ds1_file.nc")
    save_dataset(ds2, tmp_path / "ds2_file.nc")
    return tmp_path, ds1, ds2


def test_dataset_mapping():
    """Basic mapping functionality of ``DatasetMapping``."""
    ds_mapping = DatasetMapping()

    ds_mapping["ds1"] = xr.DataArray([1, 2]).to_dataset(name="data")
    ds_mapping["ds2"] = xr.DataArray([3, 4]).to_dataset(name="data")

    assert "ds1" in ds_mapping
    assert "ds2" in ds_mapping
    assert len(ds_mapping) == 2

    for ds_name, expected_ds_name in zip(ds_mapping, ["ds1", "ds2"]):
        assert ds_name == expected_ds_name

    del ds_mapping["ds1"]

    assert "ds1" not in ds_mapping
    assert "ds2" in ds_mapping
    assert len(ds_mapping) == 1


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
