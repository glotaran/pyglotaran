from __future__ import annotations

from typing import Literal

from glotaran.model.item import Item
from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem
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
