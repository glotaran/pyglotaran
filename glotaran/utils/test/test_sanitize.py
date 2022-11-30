from __future__ import annotations

from typing import Any
from typing import NamedTuple

import numpy as np
import pytest

from glotaran.utils.sanitize import pretty_format_numerical
from glotaran.utils.sanitize import sanitize_list_with_broken_tuples


class MangledListTestData(NamedTuple):
    input: list[Any]
    input_sanitized: list[str] | str
    output: list[str]


test_data_list = [
    MangledListTestData(
        ["(3", 100, "13)", "(4.0", "-200)"],
        "[(3, 100, 13), (4.0, -200)]",
        ["(3, 100, 13)", "(4.0, -200)"],
    ),
    MangledListTestData(
        [(3, 100, 13), 5.5, (4.0, -200), 5.6],
        "[(3, 100, 13), 5.5, (4.0, -200), 5.6]",
        ["(3, 100, 13)", "5.5", "(4.0, -200)", "5.6"],
    ),
    MangledListTestData(
        [(3, 100, 13), -5.5, (4.0, -200), +5.6],
        "[(3, 100, 13), -5.5, (4.0, -200), 5.6]",
        ["(3, 100, 13)", "-5.5", "(4.0, -200)", "5.6"],
    ),
    MangledListTestData(
        ["(3", -100, "13)", "(4.0", "-200)"],
        "[(3, -100, 13), (4.0, -200)]",
        ["(3, -100, 13)", "(4.0, -200)"],
    ),
    MangledListTestData(
        ["(3", 100, "13)", "(4.0", "+200)"],
        "[(3, 100, 13), (4.0, +200)]",
        ["(3, 100, 13)", "(4.0, +200)"],
    ),
]


@pytest.mark.parametrize("test_data", test_data_list)
def test_mangled_list_sanitization(test_data: MangledListTestData):
    assert test_data.input_sanitized == str(test_data.input).replace("'", "")


@pytest.mark.parametrize("test_data", test_data_list)
def test_fix_tuple_string_list(test_data: MangledListTestData):
    actual = sanitize_list_with_broken_tuples(test_data.input)
    assert all(a in b for a, b in zip(actual, test_data.output))


@pytest.mark.parametrize(
    "value, decimal_places, expected",
    (
        (0.00000001, 1, "1.0e-08"),
        (-0.00000001, 1, "-1.0e-08"),
        (0.1, 1, "0.1"),
        (1.7, 1, "1.7"),
        (10, 1, "10"),
        (1.0000000000000002, 10, "1"),
        (-1.0000000000000002, 10, "-1"),
        (10, 10, "10"),
        (-10, 10, "-10"),
        (0.00000001, 8, "0.00000001"),
        (-0.00000001, 8, "-0.00000001"),
        (0.009, 2, "9.00e-03"),
        (-0.009, 2, "-9.00e-03"),
        (0.01, 2, "0.01"),
        (12.3, 2, "12.30"),
        (np.nan, 1, "nan"),
        (np.inf, 1, "inf"),
        (-np.inf, 1, "-inf"),
    ),
)
def test_pretty_format_numerical(value: float, decimal_places: int, expected: str):
    """Pretty format values depending on decimal_places to show."""
    result = pretty_format_numerical(value, decimal_places)

    assert result == expected


if __name__ == "__main__":
    for test_data in test_data_list:
        test_mangled_list_sanitization(test_data)
        test_fix_tuple_string_list(test_data)
