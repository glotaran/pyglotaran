from glotaran.model.item_new import Attribute
from glotaran.model.item_new import Item
from glotaran.model.item_new import LibraryItem
from glotaran.model.item_new import LibraryItemType
from glotaran.model.item_new import ParameterType
from glotaran.model.item_new import TypedItem
from glotaran.model.item_new import get_structure_and_type_from_field
from glotaran.model.item_new import iterate_library_item_fields
from glotaran.model.item_new import iterate_parameter_fields
from glotaran.parameter import Parameter


class MockLibraryItem(LibraryItem):
    """A library item for testing."""

    test_attr: str = Attribute(description="Test description.")


class MockItem(Item):
    cscalar: int
    cscalar_option: int | None
    clist: list[int]
    clist_option: list[int] | None
    cdict: dict[str, int]
    cdict_option: dict[str, int] | None
    iscalar: LibraryItemType[MockLibraryItem]
    iscalar_option: LibraryItemType[MockLibraryItem] | None
    ilist: list[LibraryItemType[MockLibraryItem]]
    ilist_option: list[LibraryItemType[MockLibraryItem]] | None
    idict: dict[str, LibraryItemType[MockLibraryItem]]
    idict_option: dict[str, LibraryItemType[MockLibraryItem]] | None
    pscalar: ParameterType
    pscalar_option: ParameterType | None
    plist: list[ParameterType]
    plist_option: list[ParameterType] | None
    pdict: dict[str, ParameterType]
    pdict_option: dict[str, ParameterType] | None


class MockTypedItem(TypedItem):
    pass


class MockTypedItemConcrete(MockTypedItem):
    type: str = "concrete_type"


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
            (None, MockLibraryItem),
            (None, MockLibraryItem),
            (list, MockLibraryItem),
            (list, MockLibraryItem),
            (dict, MockLibraryItem),
            (dict, MockLibraryItem),
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
    assert MockTypedItem.get_item_types() == [MockTypedItemConcrete.get_item_type()]
    assert (
        MockTypedItem.get_item_type_class(MockTypedItemConcrete.get_item_type())
        is MockTypedItemConcrete
    )


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
                "factory": None,
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
                "factory": None,
                "type": "string",
            },
            "test_attr": {
                "title": "Test Attr",
                "description": "Test description.",
                "factory": None,
                "type": "string",
            },
        },
        "required": ["label", "test_attr"],
        "additionalProperties": False,
    }

    print(got)
    assert got == wanted
