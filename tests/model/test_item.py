from __future__ import annotations

import random
from typing import Literal

from glotaran.model.item import Item
from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem
from glotaran.model.item import add_to_initial
from glotaran.model.item import get_item_issues
from glotaran.model.item import get_structure_and_type_from_field
from glotaran.model.item import iterate_parameter_fields
from glotaran.model.item import resolve_item_parameters
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


class MockItem(Item):
    cscalar: int
    cscalar_option: int | None = None
    clist: list[int]
    clist_option: list[int] | None = None
    cdict: dict[str, int]
    cdict_option: dict[str, int] | None = None
    pscalar: ParameterType
    pscalar_option: ParameterType | None = None
    plist: list[ParameterType]
    plist_option: list[ParameterType] | None = None
    pdict: dict[str, ParameterType]
    pdict_option: dict[str, ParameterType] | None = None


class MockTypedItem(TypedItem):
    """This is just a mock item for testing."""


class MockTypedItemConcrete1(MockTypedItem):
    type: Literal["concrete_type1"]
    vint: int


class MockTypedItemConcrete2(MockTypedItem):
    type: Literal["concrete_type2"]
    vstring: str


def test_item_fields_structures_and_type():
    item_fields = MockItem.model_fields.values()
    wanted = (
        (None, int),
        (None, int),
        (list, int),
        (list, int),
        (dict, int),
        (dict, int),
        # Actual parameters instead of const values
        (None, Parameter),
        (None, Parameter),
        (list, Parameter),
        (list, Parameter),
        (dict, Parameter),
        (dict, Parameter),
    )

    assert len(item_fields) == len(wanted)
    for field, field_wanted in zip(item_fields, wanted):
        assert get_structure_and_type_from_field(field) == field_wanted


def test_iterate_parameters():
    item_fields = list(iterate_parameter_fields(MockItem))
    assert len(item_fields) == 6
    assert [name for name, _ in item_fields] == [
        "pscalar",
        "pscalar_option",
        "plist",
        "plist_option",
        "pdict",
        "pdict_option",
    ]


def test_typed_item():
    assert MockTypedItem.__item_types__ == [MockTypedItemConcrete1, MockTypedItemConcrete2]


def test_get_issues():
    item = MockItem(
        cscalar=0,
        clist=[],
        cdict={},
        pscalar="foo",
        plist=["foo", "bar"],
        pdict={"1": "foo", "2": "bar"},
    )

    issues = get_item_issues(item, Parameters({}))
    assert len(issues) == 5


def test_resolve_item_parameters():
    item = MockItem(
        cscalar=2,
        clist=[],
        cdict={},
        pscalar="param1",
        plist=["param1", "param2"],
        pdict={"foo": "param2"},
    )

    parameters = Parameters({})
    initial = Parameters.from_list(
        [
            ["param1", 1.0],
            ["param2", 2.0],
        ]
    )

    resolved = resolve_item_parameters(item, parameters, initial)
    assert item is not resolved

    assert parameters.has("param1")
    assert parameters.get("param1") is not initial.get("param1")
    assert parameters.has("param2")
    assert parameters.get("param2") is not initial.get("param2")

    assert isinstance(resolved.pscalar, Parameter)
    assert resolved.pscalar.value == 1.0

    assert isinstance(resolved.plist[0], Parameter)
    assert resolved.plist[0].value == 1.0
    assert isinstance(resolved.plist[1], Parameter)
    assert resolved.plist[1].value == 2.0

    assert isinstance(resolved.pdict["foo"], Parameter)
    assert resolved.pdict["foo"].value == 2.0

    parameters.get("param1").value = 10
    assert parameters.get("param1").value != initial.get("param1").value
    assert resolved.pscalar.value == 10


RATES_DICT = {
    "rates": [
        ["k1sum", 10, {"vary": False, "non-negative": True}],
        ["k1", 2.0, {"expr": "$b.1 * $rates.k1sum", "non-negative": True, "vary": False}],
        ["k2", 8.0, {"expr": "$b.2 * $rates.k1sum", "non-negative": True, "vary": False}],
        ["k3", 0.54321, {"non-negative": True}],
    ]
}

B_DICT = {
    "b": [
        ["1", 0.2, {"vary": True, "non-negative": True}],
        ["2", 0.8, {"expr": "1.0 - $b.1", "non-negative": True, "vary": False}],
    ]
}


def _add_parameters_to_initial(labels, parameters, initial_parameters):
    for label in labels:
        add_to_initial(label, parameters, initial_parameters)


def _test_add_to_initial(initial_parameters):
    def check_parameter_order(parameters, expected_order):
        actual_order = list(parameters._parameters.keys())
        assert (
            actual_order == expected_order
        ), f"Parameters not in expected order. Got: {actual_order}, Expected: {expected_order}"

    def test_ordered_addition():
        parameters = Parameters({})
        _add_parameters_to_initial(
            ["b.1", "b.2", "rates.k1", "rates.k2"], parameters, initial_parameters
        )
        check_parameter_order(parameters, ["b.1", "b.2", "rates.k1sum", "rates.k1", "rates.k2"])

    def test_unordered_addition():
        parameters = Parameters({})
        _add_parameters_to_initial(
            ["b.2", "b.1", "rates.k1", "rates.k2"], parameters, initial_parameters
        )
        check_parameter_order(parameters, ["b.1", "b.2", "rates.k1sum", "rates.k1", "rates.k2"])

    def test_partial_addition():
        parameters = Parameters({})
        _add_parameters_to_initial(["rates.k1", "rates.k2"], parameters, initial_parameters)
        check_parameter_order(parameters, ["b.1", "rates.k1sum", "rates.k1", "b.2", "rates.k2"])

    def test_randomized_addition():
        parameter_labels = ["b.1", "b.2", "rates.k1", "rates.k2", "rates.k3"]
        random.shuffle(parameter_labels)
        parameters = Parameters({})
        for label in parameter_labels:
            add_to_initial(label, parameters, initial_parameters)

        keys_list = list(parameters._parameters.keys())

        assert keys_list.index("rates.k2") > keys_list.index(
            "b.2"
        ), "rates.k2 should come after b.2"
        assert keys_list.index("b.2") > keys_list.index("b.1"), "b.2 should come after b.1"
        assert keys_list.index("rates.k1") > keys_list.index(
            "rates.k1sum"
        ), "rates.k1 should come after rates.k1sum"
        assert keys_list.index("rates.k2") > keys_list.index(
            "rates.k1sum"
        ), "rates.k2 should come after rates.k1sum"

        return parameters

    def check_expressions(parameters):
        assert (
            parameters.get("b.2").expression == "1.0 - $b.1"
        ), f"Incorrect expression for b.2. Got: {parameters.get('b.2').expression}"
        assert (
            parameters.get("rates.k1").expression == "$b.1 * $rates.k1sum"
        ), f"Incorrect expression for rates.k1. Got: {parameters.get('rates.k1').expression}"
        assert (
            parameters.get("rates.k2").expression == "$b.2 * $rates.k1sum"
        ), f"Incorrect expression for rates.k2. Got: {parameters.get('rates.k2').expression}"

    test_ordered_addition()
    test_unordered_addition()
    test_partial_addition()
    parameters = test_randomized_addition()
    check_expressions(parameters)


def test_add_to_initial_b_first():
    return _test_add_to_initial(Parameters.from_dict({**B_DICT, **RATES_DICT}))


def test_add_to_initial_b_last():
    return _test_add_to_initial(Parameters.from_dict({**RATES_DICT, **B_DICT}))


if __name__ == "__main__":
    for _ in range(1000):
        test_add_to_initial_b_first()
        test_add_to_initial_b_last()
