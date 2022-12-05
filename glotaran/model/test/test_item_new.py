from __future__ import annotations

from typing import Literal

from glotaran.model.item_new import Attribute
from glotaran.model.item_new import Item
from glotaran.model.item_new import LibraryItem
from glotaran.model.item_new import LibraryItemType
from glotaran.model.item_new import LibraryItemTyped
from glotaran.model.item_new import ParameterType
from glotaran.model.item_new import get_structure_and_type_from_field
from glotaran.model.item_new import iterate_library_item_fields
from glotaran.model.item_new import iterate_parameter_fields
from glotaran.parameter import Parameter


class MockLibraryItem(LibraryItem):
    """A library item for testing."""

    test_attr: str = Attribute(description="Test description.")


class MockLibraryItemNested(LibraryItem):
    """A library item for testing."""

    test_reference: LibraryItemType[MockLibraryItem] | None = None


class MockItem(Item):
    cscalar: int
    cscalar_option: int | None
    clist: list[int]
    clist_option: list[int] | None
    cdict: dict[str, int]
    cdict_option: dict[str, int] | None
    iscalar: LibraryItemType[MockLibraryItemNested]
    iscalar_option: LibraryItemType[MockLibraryItemNested] | None
    ilist: list[LibraryItemType[MockLibraryItemNested]]
    ilist_option: list[LibraryItemType[MockLibraryItemNested]] | None
    idict: dict[str, LibraryItemType[MockLibraryItemNested]]
    idict_option: dict[str, LibraryItemType[MockLibraryItemNested]] | None
    pscalar: ParameterType
    pscalar_option: ParameterType | None
    plist: list[ParameterType]
    plist_option: list[ParameterType] | None
    pdict: dict[str, ParameterType]
    pdict_option: dict[str, ParameterType] | None


class MockTypedItem(LibraryItemTyped):
    pass


class MockTypedItemConcrete1(MockTypedItem):
    type: Literal["concrete_type1"]
    vint: int


class MockTypedItemConcrete2(MockTypedItem):
    type: Literal["concrete_type2"]
    vstring: str


def test_item_fields_structures_and_type():
    item_fields = MockItem.__fields__.values()
    wanted = (
        (
            (None, int),
            (None, int),
            (list, int),
            (list, int),
            (dict, int),
            (dict, int),
        )
        + (
            (None, MockLibraryItemNested),
            (None, MockLibraryItemNested),
            (list, MockLibraryItemNested),
            (list, MockLibraryItemNested),
            (dict, MockLibraryItemNested),
            (dict, MockLibraryItemNested),
        )
        + (
            (None, Parameter),
            (None, Parameter),
            (list, Parameter),
            (list, Parameter),
            (dict, Parameter),
            (dict, Parameter),
        )
    )

    assert len(item_fields) == len(wanted)
    for field, field_wanted in zip(item_fields, wanted):
        assert get_structure_and_type_from_field(field) == field_wanted


def test_iterate_library_items():
    item_fields = list(iterate_library_item_fields(MockLibraryItemNested))
    assert len(item_fields) == 1
    item_fields = list(iterate_library_item_fields(MockItem))
    assert len(item_fields) == 6
    assert [i.name for i in item_fields] == [
        "iscalar",
        "iscalar_option",
        "ilist",
        "ilist_option",
        "idict",
        "idict_option",
    ]


def test_iterate_parameters():
    item_fields = list(iterate_parameter_fields(MockItem))
    assert len(item_fields) == 6
    assert [i.name for i in item_fields] == [
        "pscalar",
        "pscalar_option",
        "plist",
        "plist_option",
        "pdict",
        "pdict_option",
    ]


def test_typed_item():
    assert MockTypedItem.__item_types__ == [MockTypedItemConcrete1, MockTypedItemConcrete2]


def test_item_schema():
    got = LibraryItem.schema()
    wanted = {
        "title": "LibraryItem",
        "description": "An item with a label.",
        "type": "object",
        "properties": {
            "label": {
                "title": "Label",
                "description": "The label of the library item.",
                "type": "string",
            }
        },
        "required": ["label"],
        "additionalProperties": False,
    }

    print(got)
    assert got == wanted

    got = MockLibraryItem.schema()
    wanted = {
        "title": "MockLibraryItem",
        "description": "A library item for testing.",
        "type": "object",
        "properties": {
            "label": {
                "title": "Label",
                "description": "The label of the library item.",
                "type": "string",
            },
            "test_attr": {
                "title": "Test Attr",
                "description": "Test description.",
                "type": "string",
            },
        },
        "required": ["label", "test_attr"],
        "additionalProperties": False,
    }

    print(got)
    assert got == wanted


def test_get_library_name():
    assert MockLibraryItem.get_library_name() == "mock_library_item"
