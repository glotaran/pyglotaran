from __future__ import annotations

import pickle
import re
from typing import Any

import numpy as np
import pytest

from glotaran.parameter import Parameter


@pytest.mark.parametrize(
    "key_name, value_1, value_2",
    (
        ("value", 1, 2),
        ("vary", True, False),
        ("minimum", -np.inf, -1),
        ("maximum", np.inf, 1),
        ("expression", None, "$a.1*10"),
        ("standard_error", np.nan, 1),
        ("non_negative", True, False),
    ),
)
def test_parameter__deep_equals(key_name: str, value_1: Any, value_2: Any):
    parameter_1 = Parameter(label="foo", **{key_name: value_1})
    parameter_2 = Parameter(label="foo", **{key_name: value_1})
    assert parameter_1._deep_equals(parameter_2)

    parameter_3 = Parameter(label="foo", **{key_name: value_1})
    parameter_4 = Parameter(label="foo", **{key_name: value_2})
    assert not parameter_3._deep_equals(parameter_4)


@pytest.mark.parametrize("label, expected", (("foo", "foo"), (0, "0"), (1, "1")))
def test_parameter_label_always_str_or_None(label: str | int, expected: str):
    """Parameter.label is always a string"""
    parameter = Parameter(label=label)
    assert parameter.label == expected


@pytest.mark.parametrize(
    "label",
    ("exp", np.nan),
)
def test_parameter_label_error_wrong_label_pattern(label: str | int | float):
    """Error if label can't be casted to a valid label str"""
    with pytest.raises(ValueError, match=f"'{label}' is not a valid parameter label."):
        Parameter(label=label)


@pytest.mark.parametrize(
    "parameter, expected_repr",
    (
        (
            Parameter(label="foo"),
            "Parameter(label='foo')",
        ),
        (
            Parameter(label="foo", expression="$foo.bar", value=1.0, vary=True),
            # vary gets set to False due to the usage of expression
            "Parameter(label='foo', value=1.0, expression='$foo.bar', vary=False)",
        ),
    ),
)
def test_parameter_repr(parameter: Parameter, expected_repr: str):
    """Repr creates code to recreate the object."""
    print(parameter.__repr__())
    assert parameter.__repr__() == expected_repr
    assert parameter._deep_equals(eval(expected_repr))


def test_parameter_from_list():
    params = [["5", 1], ["4", 2], ["3", 3]]

    parameters = [Parameter.from_list(v) for v in params]

    assert [p.label for p in parameters] == [v[0] for v in params]
    assert [p.value for p in parameters] == [v[1] for v in params]


def test_parameter_options():
    params = [
        ["5", 1, {"non-negative": False, "min": -1, "max": 1, "vary": False}],
        ["6", 4e2, {"non-negative": True, "min": -7e2, "max": 8e2, "vary": True}],
        ["7", 2e4],
    ]

    parameters = [Parameter.from_list(v) for v in params]

    assert parameters[0].value == 1.0
    assert not parameters[0].non_negative
    assert parameters[0].minimum == -1
    assert parameters[0].maximum == 1
    assert not parameters[0].vary

    assert parameters[1].value == 4e2
    assert parameters[1].non_negative
    assert parameters[1].minimum == -7e2
    assert parameters[1].maximum == 8e2
    assert parameters[1].vary

    assert parameters[2].value == 2e4
    assert not parameters[2].non_negative
    assert parameters[2].minimum == float("-inf")
    assert parameters[2].maximum == float("inf")
    assert parameters[2].vary


def test_parameter_value_not_numeric_error():
    """Error if value isn't numeric."""
    with pytest.raises(TypeError):
        Parameter(label="", value="foo")


def test_parameter_maximum_not_numeric_error():
    """Error if maximum isn't numeric."""
    with pytest.raises(TypeError):
        Parameter(label="", maximum="foo")


def test_parameter_minimum_not_numeric_error():
    """Error if minimum isn't numeric."""
    with pytest.raises(TypeError):
        Parameter(label="", minimum="foo")


def test_parameter_non_negative():
    notnonneg = Parameter(label="", value=1, non_negative=False)
    valuenotnoneg, _, _ = notnonneg.get_value_and_bounds_for_optimization()
    assert np.allclose(1, valuenotnoneg)
    notnonneg.set_value_from_optimization(valuenotnoneg)
    assert np.allclose(1, notnonneg.value)

    nonneg1 = Parameter(label="", value=1, non_negative=True)
    value1, _, _ = nonneg1.get_value_and_bounds_for_optimization()
    assert not np.allclose(1, value1)
    nonneg1.set_value_from_optimization(value1)
    assert np.allclose(1, nonneg1.value)

    nonneg2 = Parameter(label="", value=2, non_negative=True)
    value2, _, _ = nonneg2.get_value_and_bounds_for_optimization()
    assert not np.allclose(2, value2)
    nonneg2.set_value_from_optimization(value2)
    assert np.allclose(2, nonneg2.value)

    nonnegminmax = Parameter(label="", value=5, minimum=3, maximum=6, non_negative=True)
    value5, minimum, maximum = nonnegminmax.get_value_and_bounds_for_optimization()
    assert not np.allclose(5, value5)
    assert not np.allclose(3, minimum)
    assert not np.allclose(6, maximum)


@pytest.mark.parametrize(
    "case",
    [
        ("$1", "parameters.get('1').value"),
        (
            "1 - $kinetic.1 * exp($kinetic.2) + $kinetic.3",
            "1 - parameters.get('kinetic.1').value * exp(parameters.get('kinetic.2').value) "
            "+ parameters.get('kinetic.3').value",
        ),
        ("2", "2"),
        (
            "1 - sum([$kinetic.1, $kinetic.2])",
            "1 - sum([parameters.get('kinetic.1').value, parameters.get('kinetic.2').value])",
        ),
        ("exp($kinetic.4)", "exp(parameters.get('kinetic.4').value)"),
        ("$kinetic.5", "parameters.get('kinetic.5').value"),
        (
            "$parameters.parameters.param1 + $kinetic6",
            "parameters.get('parameters.parameters.param1').value "
            "+ parameters.get('kinetic6').value",
        ),
        (
            "$foo.7.bar + $kinetic6",
            "parameters.get('foo.7.bar').value " "+ parameters.get('kinetic6').value",
        ),
        ("$1", "parameters.get('1').value"),
        ("$1-$2", "parameters.get('1').value-parameters.get('2').value"),
        ("$1-$5", "parameters.get('1').value-parameters.get('5').value"),
        (
            "100 - $inputs1.s1 - $inputs1.s3 - $inputs1.s8 - $inputs1.s12",
            "100 - parameters.get('inputs1.s1').value - parameters.get('inputs1.s3').value "
            "- parameters.get('inputs1.s8').value - parameters.get('inputs1.s12').value",
        ),
    ],
)
def test_transform_expression(case):
    expression, wanted_parameters = case
    parameter = Parameter(label="", expression=expression)
    assert parameter.transformed_expression == wanted_parameters
    # just for the test coverage so the if condition is wrong
    assert parameter.transformed_expression == wanted_parameters


def test_label_validator():
    valid_names = [
        "1",
        "valid1",
        "_valid2",
        "extra_valid3",
    ]
    for label in valid_names:
        Parameter(label)

    invalid_names = [
        "testé",
        "kinetic::red",
        "kinetic_blue+kinetic_red",
        "makesthissense=trueandfalse",
        "what/about\\slashes",
        "$invalid",
        "round",
        "parameters",
    ]

    for label in invalid_names:
        print(label)
        with pytest.raises(
            ValueError, match=re.escape(f"'{label}' is not a valid parameter label.")
        ):
            Parameter(label=label)


def test_parameter_pickle(tmpdir):
    parameter = Parameter(
        label="testlabel",
        expression="testexpression",
        minimum=1,
        maximum=2,
        non_negative=True,
        value=42,
        vary=False,
    )

    with open(tmpdir.join("test_param_pickle"), "wb") as f:
        pickle.dump(parameter, f)
    with open(tmpdir.join("test_param_pickle"), "rb") as f:
        pickled_parameter = pickle.load(f)

    assert parameter == pickled_parameter


def test_parameter_numpy_operations():
    """Operators work like a float"""
    parm1 = Parameter(label="foo", value=1)
    parm1_neg = Parameter(label="foo", value=-1)
    parm2 = Parameter(label="foo", value=2)
    parm3 = Parameter(label="foo", value=3)
    parm3_5 = Parameter(label="foo", value=3.5)

    assert parm1 == 1
    assert parm1 != 2
    assert -parm1 == -1
    assert abs(parm1_neg) == 1
    assert int(parm1) == 1
    assert np.allclose(float(parm1), 1)
    assert np.allclose(parm1 + parm2, 3)
    assert np.allclose(parm3 - parm2, 1)
    assert np.allclose(parm3 * parm2, 6)
    assert np.allclose(parm3 / parm2, 1.5)
    assert parm3 // parm2 == 1
    assert np.trunc(parm3_5) == 3
    assert parm2 % 2 == 0
    assert 2 % parm2 == 0
    assert divmod(parm3, 2) == (1, 1)
    assert divmod(3, parm2) == (1, 1)
    assert np.allclose(parm3**parm2, 9)
    assert parm3 >= parm2
    assert parm3 > parm2
    assert parm1 <= parm2
    assert parm1 < parm2


def test_parameter_dict_roundtrip():
    param = Parameter(
        label="foo",
        expression="1",
        maximum=2,
        minimum=1,
        non_negative=True,
        value=42,
        vary=False,
    )

    param_dict = param.as_dict()
    print(param_dict)
    param_from_dict = Parameter(**param_dict)

    assert param.label == param_from_dict.label
    assert param.expression == param_from_dict.expression
    assert param.maximum == param_from_dict.maximum
    assert param.minimum == param_from_dict.minimum
    assert param.non_negative == param_from_dict.non_negative
    assert param.value == param_from_dict.value
    assert param.vary == param_from_dict.vary


def test_parameter_list_roundtrip():
    param = Parameter(
        label="foo",
        expression="1",
        maximum=2,
        minimum=1,
        non_negative=True,
        value=42,
        vary=False,
    )

    param_list = param.as_list()
    print(param_list)
    param_from_list = Parameter.from_list(param_list)

    assert param.label == param_from_list.label
    assert param.expression == param_from_list.expression
    assert param.maximum == param_from_list.maximum
    assert param.minimum == param_from_list.minimum
    assert param.non_negative == param_from_list.non_negative
    assert param.value == param_from_list.value
    assert param.vary == param_from_list.vary
