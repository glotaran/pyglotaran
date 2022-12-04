from glotaran.model.item_new import Attribute
from glotaran.model.item_new import Item
from glotaran.model.item_new import ModelItem
from glotaran.model.item_new import ModelItemType
from glotaran.model.item_new import ParameterType
from glotaran.model.item_new import TypedItem
from glotaran.model.item_new import get_structure_and_type_from_field
from glotaran.model.item_new import iterate_model_item_fields
from glotaran.model.item_new import iterate_parameter_fields
from glotaran.parameter import Parameter


class MockModelItem(ModelItem):
    """A model item for testing."""

    test_attr: str = Attribute(description="Test description.")


class MockItem(Item):
    cscalar: int
    cscalar_option: int | None
    clist: list[int]
    clist_option: list[int] | None
    cdict: dict[str, int]
    cdict_option: dict[str, int] | None
    iscalar: ModelItemType[MockModelItem]
    iscalar_option: ModelItemType[MockModelItem] | None
    ilist: list[ModelItemType[MockModelItem]]
    ilist_option: list[ModelItemType[MockModelItem]] | None
    idict: dict[str, ModelItemType[MockModelItem]]
    idict_option: dict[str, ModelItemType[MockModelItem]] | None
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
            (None, MockModelItem),
            (None, MockModelItem),
            (list, MockModelItem),
            (list, MockModelItem),
            (dict, MockModelItem),
            (dict, MockModelItem),
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


def test_iterate_model_items():
    item_fields = list(iterate_model_item_fields(MockItem))
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
    got = ModelItem.schema()
    wanted = {
        "title": "ModelItem",
        "description": "An item with a label.",
        "type": "object",
        "properties": {
            "label": {
                "title": "Label",
                "description": "The label of the model item.",
                "factory": None,
                "type": "string",
            }
        },
        "required": ["label"],
    }

    assert got == wanted

    got = MockModelItem.schema()
    wanted = {
        "title": "MockModelItem",
        "description": "A model item for testing.",
        "type": "object",
        "properties": {
            "label": {
                "title": "Label",
                "description": "The label of the model item.",
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
    }

    print(got)
    assert got == wanted
