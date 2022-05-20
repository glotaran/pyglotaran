from __future__ import annotations

from textwrap import dedent

import numpy as np

from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.parameter import ParameterGroup


def test_parameter_group_copy():
    params = """
    kinetic:
        - ["5", 1, {non-negative: true, min: -1, max: 1, vary: false}]
        - 4
        - 5
    j:
        - 7
        - 8
    """
    params = load_parameters(params, format_name="yml_str")
    copy = params.copy()

    for label, parameter in params.all():
        assert copy.has(label)
        copied_parameter = copy.get(label)
        assert parameter.value == copied_parameter.value
        assert parameter.non_negative == copied_parameter.non_negative
        assert parameter.minimum == copied_parameter.minimum
        assert parameter.maximum == copied_parameter.maximum
        assert parameter.vary == copied_parameter.vary


def test_parameter_group_from_list():
    params = """
    - 5
    - 4
    - 3
    - 2
    - 1
    """

    params = load_parameters(params, format_name="yml_str")

    assert len(list(params.all())) == 5

    assert [p.label for _, p in params.all()] == [f"{i}" for i in range(1, 6)]
    assert [p.value for _, p in params.all()] == list(range(1, 6))[::-1]


def test_parameter_group_from_dict():
    params = """
    kinetic:
        - 3
        - 4
        - 5
    j:
        - 7
        - 8
    """

    params = load_parameters(params, format_name="yml_str")

    assert len(list(params.all())) == 5
    group = params["kinetic"]
    assert len(list(group.all())) == 3
    assert [p.label for _, p in group.all()] == [f"{i}" for i in range(1, 4)]
    assert [p.value for _, p in group.all()] == list(range(3, 6))
    group = params["j"]
    assert len(list(group.all())) == 2
    assert [p.label for _, p in group.all()] == [f"{i}" for i in range(1, 3)]
    assert [p.value for _, p in group.all()] == list(range(7, 9))


def test_parameter_group_from_dict_nested():
    params = """
    kinetic:
        j:
            - 7
            - 8
            - 9
    """

    params = load_parameters(params, format_name="yml_str")
    assert len(list(params.all())) == 3
    group = params["kinetic"]
    assert len(list(group.all())) == 3
    group = group["j"]
    assert len(list(group.all())) == 3
    assert [p.label for _, p in group.all()] == [f"{i}" for i in range(1, 4)]
    assert [p.value for _, p in group.all()] == list(range(7, 10))

    assert params.get("kinetic.j.1").full_label == "kinetic.j.1"

    roundtrip_df = ParameterGroup.from_dataframe(params.to_dataframe()).to_dataframe()
    assert all(roundtrip_df.label == params.to_dataframe().label)


def test_parameter_group_to_array():
    params = """
    - ["1", 1, {non-negative: false, min: -1, max: 1, vary: false}]
    - ["2", 4e2, {non-negative: true, min: 10, max: 8e2, vary: true}]
    - ["3", 2e4]
    """

    params = load_parameters(params, format_name="yml_str")

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
    assert len(upper_bounds_only_vary) == 2

    assert labels_only_vary == ["2", "3"]


def test_parameter_group_set_from_label_and_value_arrays():
    params = """
    - ["1", 1, {non-negative: false, min: -1, max: 1, vary: false}]
    - ["2", 4e2, {non-negative: true, min: 10, max: 8e2, vary: true}]
    - ["3", 2e4]
    """

    params = load_parameters(params, format_name="yml_str")

    labels = ["1", "2", "3"]
    values = [0, np.log(6e2), 42]

    params.set_from_label_and_value_arrays(labels, values)

    values[1] = np.exp(values[1])

    for i in range(3):
        assert params.get(f"{i+1}").value == values[i]


def test_parameter_group_from_csv(tmpdir):

    TEST_CSV = dedent(
        """\
        label, value, minimum, maximum, vary, non-negative, expression
        rates.k1,0.050,0,5,True,True,None
        rates.k2,None,,,True,True,$rates.k1 * 2
        rates.k3,2.311,,,True,True,None
        pen.eq.1,1.000,,,False,False,None
        """
    )

    csv_path = tmpdir.join("parameters.csv")
    with open(csv_path, "w") as f:
        f.write(TEST_CSV)

    params = load_parameters(csv_path)

    assert "rates" in params

    assert params.has("rates.k1")
    p = params.get("rates.k1")
    assert p.label == "k1"
    assert p.value == 0.05
    assert p.minimum == 0
    assert p.maximum == 5
    assert p.vary
    assert p.non_negative
    assert p.expression is None

    assert params.has("rates.k2")
    p = params.get("rates.k2")
    assert p.label == "k2"
    assert p.value == params.get("rates.k1") * 2
    assert p.minimum == -np.inf
    assert p.maximum == np.inf
    assert not p.vary
    assert not p.non_negative
    assert p.expression == "$rates.k1 * 2"

    assert params.has("rates.k3")
    p = params.get("rates.k3")
    assert p.label == "k3"
    assert p.value == 2.311
    assert p.minimum == -np.inf
    assert p.maximum == np.inf
    assert p.vary
    assert p.non_negative
    assert p.expression is None

    assert "pen" in params
    assert "eq" in params["pen"]

    assert params.has("pen.eq.1")
    p = params.get("pen.eq.1")
    assert p.label == "1"
    assert p.value == 1.0
    assert p.minimum == -np.inf
    assert p.maximum == np.inf
    assert not p.vary
    assert not p.non_negative
    assert p.expression is None


def test_parameter_group_to_csv(tmpdir):
    csv_path = tmpdir.join("parameters.csv")
    params = load_parameters(
        """
    b:
        - ["1", 0.25, {vary: false, min: 0, max: 8}]
        - ["2", 0.75, {expr: '1 - $b.1', non-negative: true}]
    rates:
        - ["total", 2]
        - ["branch1", {expr: '$rates.total * $b.1'}]
        - ["branch2", {expr: '$rates.total * $b.2'}]
    """,
        format_name="yml_str",
    )
    for _, p in params.all():
        p.standard_error = 42

    save_parameters(params, csv_path, "csv")
    wanted = dedent(
        """\
        label,value,expression,minimum,maximum,non-negative,vary,standard-error
        b.1,0.25,None,0.0,8.0,False,False,42
        b.2,0.75,1 - $b.1,,,False,False,42
        rates.total,2.0,None,,,False,True,42
        rates.branch1,0.5,$rates.total * $b.1,,,False,False,42
        rates.branch2,1.5,$rates.total * $b.2,,,False,False,42
        """
    )

    with open(csv_path) as f:
        got = f.read()
        print(got)
        assert got == wanted
    params_from_csv = load_parameters(csv_path)

    for label, p in params.all():
        assert params_from_csv.has(label)
        p_from_csv = params_from_csv.get(label)
        assert p.label == p_from_csv.label
        assert p.value == p_from_csv.value
        assert p.minimum == p_from_csv.minimum
        assert p.maximum == p_from_csv.maximum
        assert p.vary == p_from_csv.vary
        assert p.non_negative == p_from_csv.non_negative
        assert p.expression == p_from_csv.expression


def test_parameter_group_to_from_parameter_dict_list():
    parameter_group = load_parameters(
        """
    b:
        - ["1", 0.25, {vary: false, min: 0, max: 8}]
        - ["2", 0.75, {expr: '1 - $b.1', non-negative: true}]
    rates:
        - ["total", 2]
        - ["branch1", {expr: '$rates.total * $b.1'}]
        - ["branch2", {expr: '$rates.total * $b.2'}]
    """,
        format_name="yml_str",
    )

    parameter_dict_list = parameter_group.to_parameter_dict_list()
    parameter_group_from_dict_list = ParameterGroup.from_parameter_dict_list(parameter_dict_list)

    for label, wanted in parameter_group.all():
        got = parameter_group_from_dict_list.get(label)

        assert got.label == wanted.label
        assert got.full_label == wanted.full_label
        assert got.expression == wanted.expression
        assert got.maximum == wanted.maximum
        assert got.minimum == wanted.minimum
        assert got.non_negative == wanted.non_negative
        assert got.value == wanted.value
        assert got.vary == wanted.vary


def test_parameter_group_to_from_df():
    parameter_group = load_parameters(
        """
    b:
        - ["1", 0.25, {vary: false, min: 0, max: 8}]
        - ["2", 0.75, {expr: '1 - $b.1', non-negative: true}]
    rates:
        - ["total", 2]
        - ["branch1", {expr: '$rates.total * $b.1'}]
        - ["branch2", {expr: '$rates.total * $b.2'}]
    """,
        format_name="yml_str",
    )

    for _, p in parameter_group.all():
        p.standard_error = 42

    parameter_df = parameter_group.to_dataframe()

    for column in [
        "label",
        "value",
        "standard-error",
        "expression",
        "minimum",
        "maximum",
        "non-negative",
        "vary",
    ]:
        assert column in parameter_df

    assert all(parameter_df["standard-error"] == 42)

    parameter_group_from_df = ParameterGroup.from_dataframe(parameter_df)

    for label, wanted in parameter_group.all():
        got = parameter_group_from_df.get(label)

        assert got.label == wanted.label
        assert got.full_label == wanted.full_label
        assert got.expression == wanted.expression
        assert got.maximum == wanted.maximum
        assert got.minimum == wanted.minimum
        assert got.non_negative == wanted.non_negative
        assert got.value == wanted.value
        assert got.vary == wanted.vary


def test_missing_parameter_value_labels():
    """Full labels of all parameters with missing values (NaN) get listed."""
    parameter_group = load_parameters(
        dedent(
            """\
            b:
                - ["missing_value_1",]
                - ["missing_value_2"]
                - ["2", 0.75]
            kinetic:
                j:
                    - ["missing_value_3"]
            """
        ),
        format_name="yml_str",
    )

    assert parameter_group.missing_parameter_value_labels == [
        "b.missing_value_1",
        "b.missing_value_2",
        "kinetic.j.missing_value_3",
    ]
