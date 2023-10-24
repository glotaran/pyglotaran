from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


def test_parameters_from_list():
    params = [5, 4, 3, 2, 1]

    parameters = Parameters.from_list(params)

    print(parameters._parameters)
    assert len(list(parameters.all())) == 5

    assert [p.label for p in parameters.all()] == [f"{i}" for i in range(1, 6)]
    assert [p.value for p in parameters.all()] == list(range(1, 6))[::-1]


def test_parameters_from_dict():
    params = {"a": [3, 4, 5], "b": [7, 8]}

    parameters = Parameters.from_dict(params)

    assert len(list(parameters.all())) == 5

    for label, value in [
        ("a.1", 3),
        ("a.2", 4),
        ("a.3", 5),
        ("b.1", 7),
        ("b.2", 8),
    ]:
        assert parameters.has(label)
        assert parameters.get(label).label == label
        assert parameters.get(label).value == value


def test_parameters_from_dict_nested():
    params = {"a": {"b": [7, 8, 9]}}

    parameters = Parameters.from_dict(params)
    assert len(list(parameters.all())) == 3

    for label, value in [
        ("a.b.1", 7),
        ("a.b.2", 8),
        ("a.b.3", 9),
    ]:
        assert parameters.has(label)
        assert parameters.get(label).label == label
        assert parameters.get(label).value == value


def test_parameters_default_options():
    params = {"block": [1.0, [3.4, {"vary": True}], {"vary": False}]}

    parameters = Parameters.from_dict(params)
    assert len(list(parameters.all())) == 2

    assert not parameters.get("block.1").vary
    assert parameters.get("block.2").vary


def test_parameter_group_to_from_parameter_dict_list():
    parameters = Parameters.from_dict(
        {
            "a": [
                ["1", 0.25, {"vary": False, "min": 0, "max": 8}],
                ["2", 0.75, {"expr": "1 - $a.1", "non-negative": True}],
            ],
            "b": [
                ["total", 2],
                ["branch1", {"expr": "$b.total * $a.1"}],
                ["branch2", {"expr": "ln($b.total) * $a.2"}],
            ],
        }
    )

    parameters_dict_list = parameters.to_parameter_dict_list()

    assert parameters == Parameters.from_parameter_dict_list(parameters_dict_list)


def test_parameters_equal():
    """Instances of ``Parameters`` that have the same values are equal."""
    params = [2, 1]

    parameters_1 = Parameters.from_list(params)
    parameters_2 = Parameters.from_list(params)

    assert parameters_1 == parameters_2


@pytest.mark.parametrize(
    "key_name, value_1, value_2",
    (
        ("vary", True, False),
        ("min", -np.inf, -1),
        ("max", np.inf, 1),
        ("expression", None, "$a.1*10"),
        ("standard-error", np.nan, 1),
        ("non-negative", True, False),
    ),
)
def test_parameters_not_equal(key_name: str, value_1: Any, value_2: Any):
    """Instances of ``Parameters`` that have the same values are equal."""
    parameters_1 = Parameters.from_dict({"a": [["1", 0.25, {key_name: value_1}]]})
    parameters_2 = Parameters.from_dict({"a": [["1", 0.25, {key_name: value_2}]]})

    assert parameters_1 != parameters_2


def test_parameters_equal_error():
    """Raise if rhs operator is not an instance of ``Parameters``."""
    param_dict = {"foo": Parameter(label="foo")}
    with pytest.raises(NotImplementedError) as excinfo:
        Parameters(param_dict) == param_dict

    assert (
        str(excinfo.value)
        == "Parameters can only be compared with instances of Parameters, not with 'dict'."
    )


def test_parameter_scientific_values():
    values = [5e3, -4.2e-4, 3e-2, -2e6]
    assert [p.value for p in Parameters.from_list(values).all()] == values


def test_parameter_group_copy():
    parameters = Parameters.from_dict(
        {
            "a": [
                ["1", 0.25, {"vary": False, "min": 0, "max": 8}],
                ["2", 0.75, {"expr": "1 - $a.1", "non-negative": True}],
            ],
            "b": [
                ["total", 2],
                ["branch1", {"expr": "$b.total * $a.1"}],
                ["branch2", {"expr": "$b.total * $a.2"}],
            ],
        }
    )

    copy = parameters.copy()

    assert parameters is not copy
    assert parameters == parameters.copy()


def test_parameter_expressions():
    parameters = Parameters.from_list(
        [["1", 2], ["2", 5], ["3", {"expr": "$1 * exp($2)"}], ["4", {"expr": "2"}]]
    )

    assert parameters.get("3").expression is not None
    assert not parameters.get("3").vary
    assert parameters.get("3").value == 2 * np.exp(5)
    assert parameters.get("3").value == parameters.get("1") * np.exp(parameters.get("2"))
    assert parameters.get("4").value == 2

    with pytest.raises(ValueError):
        Parameters.from_list([["3", {"expr": "None"}]])


def test_parameters_array_conversion():
    parameters = Parameters.from_list(
        [
            ["1", 1, {"non-negative": False, "min": -1, "max": 1, "vary": False}],
            ["2", 4e2, {"non-negative": True, "min": 10, "max": 8e2, "vary": True}],
            ["3", 2e4],
        ]
    )

    labels, values, lower_bounds, upper_bounds = parameters.get_label_value_and_bounds_arrays(
        exclude_non_vary=False
    )

    assert len(labels) == 3
    assert len(values) == 3
    assert len(lower_bounds) == 3
    assert len(upper_bounds) == 3

    assert labels == ["1", "2", "3"]
    assert np.allclose(values, [1, np.log(4e2), 2e4])
    assert np.allclose(lower_bounds, [-1, np.log(10), -np.inf])
    assert np.allclose(upper_bounds, [1, np.log(8e2), np.inf])

    (
        labels_only_vary,
        values_only_vary,
        lower_bounds_only_vary,
        upper_bounds_only_vary,
    ) = parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)

    assert len(labels_only_vary) == 2
    assert len(values_only_vary) == 2
    assert len(lower_bounds_only_vary) == 2
    assert len(upper_bounds_only_vary) == 2

    assert labels_only_vary == ["2", "3"]

    labels = ["1", "2", "3"]
    values = [0, np.log(6e2), 42]

    parameters.set_from_label_and_value_arrays(labels, values)

    values[1] = np.exp(values[1])

    for i in range(3):
        assert parameters.get(f"{i+1}").value == values[i]


def test_parameters_to_from_df():
    parameters = Parameters.from_dict(
        {
            "a": [
                ["1", 0.25, {"vary": False, "min": 0, "max": 8}],
                ["2", 0.75, {"expr": "1 - $a.1", "non-negative": True}],
            ],
            "b": [
                ["total", 2],
                ["branch1", {"expr": "$b.total * $a.1"}],
                ["branch2", {"expr": "$b.total * $a.2"}],
            ],
        }
    )

    for p in parameters.all():
        p.standard_error = 42

    parameter_df = parameters.to_dataframe()

    for column in [
        "label",
        "value",
        "standard_error",
        "expression",
        "minimum",
        "maximum",
        "non_negative",
        "vary",
    ]:
        assert column in parameter_df

    assert all(parameter_df["standard_error"] == 42)

    assert parameters == Parameters.from_dataframe(parameter_df)


def test_parameters_from_dataframe_minimal_required_columns():
    """No error if df only contains ``label`` and ``value`` columns and error if any is missing."""
    minimal_df = pd.DataFrame([{"label": "foo.1", "value": 1}])
    result = Parameters.from_dataframe(minimal_df)
    expected = Parameters.from_dict({"foo": [1]})

    assert result == expected

    for req_col in ["label", "value"]:
        with pytest.raises(ValueError) as exc_info:
            Parameters.from_dataframe(minimal_df.drop(req_col, axis=1))

        assert str(exc_info.value) == f"Missing required column '{req_col}' in 'DataFrame'."


@pytest.mark.parametrize(
    "column_name, expected_error_str",
    (
        ("minimum", "Column 'minimum' in 'DataFrame' has non numeric values."),
        ("maximum", "Column 'maximum' in 'DataFrame' has non numeric values."),
        ("value", "Column 'value' in 'DataFrame' has non numeric values."),
        ("non_negative", "Column 'non_negative' in 'DataFrame' has non boolean values."),
        ("vary", "Column 'vary' in 'DataFrame' has non boolean values."),
    ),
)
def test_parameters_from_dataframe(column_name: str, expected_error_str: str):
    """Check error message on bad column values."""
    minimal_df = pd.DataFrame([{"label": "foo.1", "value": 1}])
    minimal_df[column_name] = "foo"

    with pytest.raises(ValueError) as exc_info:
        Parameters.from_dataframe(minimal_df)

    assert str(exc_info.value) == expected_error_str
