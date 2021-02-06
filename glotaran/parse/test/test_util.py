from __future__ import annotations

from typing import Any
from typing import NamedTuple

import pytest

from glotaran.parse.util import sanitize_list_with_broken_tuples


class MangledListTestData(NamedTuple):
    input: list[Any]
    input_sanitized: list[str]
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


if __name__ == "__main__":
    for test_data in test_data_list:
        test_mangled_list_sanitization(test_data)
        test_fix_tuple_string_list(test_data)
