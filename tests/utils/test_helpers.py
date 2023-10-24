"""Tests for ``glotaran.utils.numeric_helpers``."""

from typing import Any

import numpy as np
import pytest

from glotaran.utils.helpers import nan_or_equal


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    (
        ("foo", "foo", True),
        (np.nan, np.nan, True),
        (1, 1, True),
        (1, 2, False),
        ("foo", "bar", False),
    ),
)
def test_nan_or_equal(lhs: Any, rhs: Any, expected: bool):
    """Only ``False`` if values actually differ."""
    assert nan_or_equal(lhs, rhs) == expected
