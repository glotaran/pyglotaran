"""Test deprecated functionality in 'glotaran.project.result'."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:
    from py.path import local as LocalPath

    from glotaran.project.result import Result


def test_Result_save_method(tmpdir: LocalPath, dummy_result: Result):  # noqa: F811
    """Result.save(result_dir) creates all file"""
    result_dir = tmpdir / "dummy"
    result_dir.mkdir()

    deprecation_warning_on_call_test_helper(
        dummy_result.save, args=[str(result_dir)], raise_exception=True
    )

    assert (result_dir / "result.md").exists()
    assert (result_dir / "optimized_parameters.csv").exists()
    assert (result_dir / "dataset1.nc").exists()
    assert (result_dir / "dataset2.nc").exists()
    assert (result_dir / "dataset3.nc").exists()


def test_Result_get_dataset_method(dummy_result: Result):  # noqa: F811
    """Result.get_dataset(dataset_label) gives correct dataset."""

    result = deprecation_warning_on_call_test_helper(
        dummy_result.get_dataset, args=["dataset1"], raise_exception=True
    )

    assert result == dummy_result.data["dataset1"]


def test_Result_get_dataset_method_error(dummy_result: Result):  # noqa: F811
    """Result.get_dataset(dataset_label) error on wrong key."""

    with pytest.raises(ValueError, match="Unknown dataset 'foo'"):
        deprecation_warning_on_call_test_helper(
            dummy_result.get_dataset, args=["foo"], raise_exception=True
        )
