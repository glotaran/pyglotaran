"""Test deprecated functionality in 'glotaran.project.result'."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:

    from glotaran.project.result import Result


def test_Result_get_dataset_method(dummy_result: Result):  # noqa: F811
    """Result.get_dataset(dataset_label) gives correct dataset."""

    _, result = deprecation_warning_on_call_test_helper(
        dummy_result.get_dataset, args=["dataset1"], raise_exception=True
    )

    assert result == dummy_result.data["dataset1"]


def test_Result_get_dataset_method_error(dummy_result: Result):  # noqa: F811
    """Result.get_dataset(dataset_label) error on wrong key."""

    with pytest.raises(ValueError, match="Unknown dataset 'foo'"):
        deprecation_warning_on_call_test_helper(
            dummy_result.get_dataset, args=["foo"], raise_exception=True
        )
