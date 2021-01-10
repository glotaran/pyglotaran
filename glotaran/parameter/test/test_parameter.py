import pickle

import numpy as np
import pytest

from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup


def test_param_array():
    params = """
    - 5
    - 4
    - 3
    - 2
    - 1
    """

    params = ParameterGroup.from_yaml(params)

    assert len(list(params.all())) == 5

    assert [p.label for _, p in params.all()] == [f"{i}" for i in range(1, 6)]
    assert [p.value for _, p in params.all()] == list(range(1, 6))[::-1]


def test_param_label():
    params = """
    - ["5", 1]
    - ["4", 2]
    - ["3", 3]
    """

    params = ParameterGroup.from_yaml(params)

    assert len(list(params.all())) == 3
    assert [p.label for _, p in params.all()] == [f"{i}" for i in range(5, 2, -1)]
    assert [p.value for _, p in params.all()] == list(range(1, 4))


def test_param_options():
    params = """
    - ["5", 1, {non-negative: false, min: -1, max: 1, vary: false}]
    - ["6", 4e2, {non-negative: true, min: -7e2, max: 8e2, vary: true}]
    - ["7", 2e4]
    """

    params = ParameterGroup.from_yaml(params)

    assert params.get("5").value == 1.0
    assert not params.get("5").non_negative
    assert params.get("5").minimum == -1
    assert params.get("5").maximum == 1
    assert not params.get("5").vary

    assert params.get("6").value == 4e2
    assert params.get("6").non_negative
    assert params.get("6").minimum == -7e2
    assert params.get("6").maximum == 8e2
    assert params.get("6").vary

    assert params.get("7").value == 2e4
    assert not params.get("7").non_negative
    assert params.get("7").minimum == float("-inf")
    assert params.get("7").maximum == float("inf")
    assert params.get("7").vary


def test_param_block_options():
    params = """
    block:
        - 1.0
        - [3.4, {vary: true}]
        - {vary: false}
    """

    params = ParameterGroup.from_yaml(params)
    assert not params.get("block.1").vary
    assert params.get("block.2").vary


def test_nested_param_list():
    params = """
    kinetic:
        - 3
        - 4
        - 5
    j:
        - 7
        - 8
    """

    params = ParameterGroup.from_yaml(params)

    assert len(list(params.all())) == 5
    group = params["kinetic"]
    assert len(list(group.all())) == 3
    assert [p.label for _, p in group.all()] == [f"{i}" for i in range(1, 4)]
    assert [p.value for _, p in group.all()] == list(range(3, 6))
    group = params["j"]
    assert len(list(group.all())) == 2
    assert [p.label for _, p in group.all()] == [f"{i}" for i in range(1, 3)]
    assert [p.value for _, p in group.all()] == list(range(7, 9))


def test_nested_param_group():
    params = """
    kinetic:
        j:
            - 7
            - 8
            - 9
    """

    params = ParameterGroup.from_yaml(params)
    assert len(list(params.all())) == 3
    group = params["kinetic"]
    assert len(list(group.all())) == 3
    group = group["j"]
    assert len(list(group.all())) == 3
    assert [p.label for _, p in group.all()] == [f"{i}" for i in range(1, 4)]
    assert [p.value for _, p in group.all()] == list(range(7, 10))


def test_non_negative():

    notnonneg = Parameter(value=1, non_negative=False)
    valuenotnoneg, _, _ = notnonneg.get_value_and_bounds_for_optimization()
    assert np.allclose(1, valuenotnoneg)
    notnonneg.set_value_from_optimization(valuenotnoneg)
    assert np.allclose(1, notnonneg.value)

    nonneg1 = Parameter(value=1, non_negative=True)
    value1, _, _ = nonneg1.get_value_and_bounds_for_optimization()
    assert not np.allclose(1, value1)
    nonneg1.set_value_from_optimization(value1)
    assert np.allclose(1, nonneg1.value)

    nonneg2 = Parameter(value=2, non_negative=True)
    value2, _, _ = nonneg2.get_value_and_bounds_for_optimization()
    assert not np.allclose(2, value2)
    nonneg2.set_value_from_optimization(value2)
    assert np.allclose(2, nonneg2.value)

    nonnegminmax = Parameter(value=5, minimum=3, maximum=6, non_negative=True)
    value5, minimum, maximum = nonnegminmax.get_value_and_bounds_for_optimization()
    assert not np.allclose(5, value5)
    assert not np.allclose(3, minimum)
    assert not np.allclose(6, maximum)


def test_parameter_group_to_array():
    params = """
    - ["1", 1, {non-negative: false, min: -1, max: 1, vary: false}]
    - ["2", 4e2, {non-negative: true, min: 10, max: 8e2, vary: true}]
    - ["3", 2e4]
    """

    params = ParameterGroup.from_yaml(params)

    labels, values, lower_bounds, upper_bounds = params.get_label_value_and_bounds_arrays(
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
    ) = params.get_label_value_and_bounds_arrays(exclude_non_vary=True)

    assert len(labels_only_vary) == 2
    assert len(values_only_vary) == 2
    assert len(lower_bounds_only_vary) == 2
    assert len(lower_bounds_only_vary) == 2

    assert labels_only_vary == ["2", "3"]


def test_update_parameter_group_from_array():
    params = """
    - ["1", 1, {non-negative: false, min: -1, max: 1, vary: false}]
    - ["2", 4e2, {non-negative: true, min: 10, max: 8e2, vary: true}]
    - ["3", 2e4]
    """

    params = ParameterGroup.from_yaml(params)

    labels = ["1", "2", "3"]
    values = [0, np.log(6e2), 42]

    params.set_from_label_and_value_arrays(labels, values)

    values[1] = np.exp(values[1])

    for i in range(3):
        assert params.get(f"{i+1}").value == values[i]


@pytest.mark.parametrize(
    "case",
    [
        ("$1", "group.get('1').value"),
        (
            "1 - $kinetic.1 * exp($kinetic.2) + $kinetic.3",
            "1 - group.get('kinetic.1').value * exp(group.get('kinetic.2').value) "
            "+ group.get('kinetic.3').value",
        ),
        ("2", "2"),
        (
            "1 - sum([$kinetic.1, $kinetic.2])",
            "1 - sum([group.get('kinetic.1').value, group.get('kinetic.2').value])",
        ),
        ("exp($kinetic.4)", "exp(group.get('kinetic.4').value)"),
        ("$kinetic.5", "group.get('kinetic.5').value"),
        (
            "$group.sub_group.param1 + $kinetic6",
            "group.get('group.sub_group.param1').value + group.get('kinetic6').value",
        ),
        ("$foo.7.bar + $kinetic6", "group.get('foo.7.bar').value + group.get('kinetic6').value"),
        ("$1", "group.get('1').value"),
        ("$1-$2", "group.get('1').value-group.get('2').value"),
        ("$1-$5", "group.get('1').value-group.get('5').value"),
    ],
)
def test_transform_expression(case):
    expression, wanted = case
    parameter = Parameter(expression=expression)
    assert parameter.transformed_expression == wanted


def test_label_validator():
    valid_names = [
        "1",
        "valid1",
        "_valid2",
        "extra_valid3",
    ]

    assert all(list(map(Parameter.valid_label, valid_names)))

    invalid_names = [
        "testé",
        "kinetic.1",
        "kinetic_red.3",
        "foo.7.bar",
        "_ilikeunderscoresatbegeninngin.justbecause",
        "42istheanswer.42",
        "kinetic::red",
        "kinetic_blue+kinetic_red",
        "makesthissense=trueandfalse",
        "what/about\\slashes",
        "$invalid",
        "round",
        "group",
    ]
    assert not any(list(map(Parameter.valid_label, invalid_names)))


def test_parameter_expressions():
    params = """
    - ["1", 2]
    - ["2", 5]
    - ["3", {expr: '$1 * exp($2)'}]
    - ["4", {expr: '2'}]
    """

    params = ParameterGroup.from_yaml(params)

    assert params.get("3").expression is not None
    assert not params.get("3").vary
    assert params.get("3").value == params.get("1") * np.exp(params.get("2"))
    assert params.get("4").value == 2

    params_bad_expr = """
    - ["3", {expr: 'None'}]
    """

    with pytest.raises(ValueError):
        ParameterGroup.from_yaml(params_bad_expr)


def test_parameter_pickle(tmpdir):

    parameter = Parameter("testlabel", "testlabelfull", "testexpression", 1, 2, True, 42, False)

    with open(tmpdir.join("test_param_pickle"), "wb") as f:
        pickle.dump(parameter, f)
    with open(tmpdir.join("test_param_pickle"), "rb") as f:
        pickled_parameter = pickle.load(f)

    assert parameter == pickled_parameter
