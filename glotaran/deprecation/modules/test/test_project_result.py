"""Test deprecated functionality in 'glotaran.project.result'."""
from __future__ import annotations

import pytest

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.optimization.optimize import optimize
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


def test_result_get_dataset_method(dummy_result: Result):
    """Result.get_dataset(dataset_label) gives correct dataset."""

    _, result = deprecation_warning_on_call_test_helper(
        dummy_result.get_dataset, args=["dataset_1"], raise_exception=True
    )

    assert result == dummy_result.data["dataset_1"]


def test_result_get_dataset_method_error(dummy_result: Result):
    """Result.get_dataset(dataset_label) error on wrong key."""

    with pytest.raises(ValueError, match="Unknown dataset 'foo'"):
        deprecation_warning_on_call_test_helper(
            dummy_result.get_dataset, args=["foo"], raise_exception=True
        )
