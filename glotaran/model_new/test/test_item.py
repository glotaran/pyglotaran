from attrs import fields

from glotaran.model_new.item import ModelItem
from glotaran.model_new.item import ModelItemType
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import item
from glotaran.model_new.item import model_items
from glotaran.model_new.item import strip_type_and_structure_from_attribute


@item
class MockModelItemDict(ModelItem):
    a_string: str
    p_scalar: ParameterType
    p_scalar_option: ParameterType | None = None


@item
class MockModelItemList:
    item1: ModelItemType[MockModelItemDict]
    item2: list[ModelItemType[MockModelItemDict]]
    item3: dict[str, ModelItemType[MockModelItemDict]]


def test_strip_type_and_structure_from_attribute():
    @item
    class MockItem:
        pscalar: int = None
        pscalar_option: int | None = None
        plist: list[int] = None
        plist_option: list[int] | None = None
        pdict: dict[str, int] = None
        pdict_option: dict[str, int] | None = None
        iscalar: ModelItemType[int] = None
        iscalar_option: ModelItemType[int] | None = None
        ilist: list[ModelItemType[int]] = None
        ilist_option: list[ModelItemType[int]] | None = None
        idict: dict[str, ModelItemType[int]] = None
        idict_option: dict[str, ModelItemType[int]] | None = None

    for attr in fields(MockItem):
        structure, type = strip_type_and_structure_from_attribute(attr)
        print(attr.name, attr.type, structure, type)
        assert structure in (None, dict, list)
        assert type is int


def test_model_get_items():
    items = list(model_items(MockModelItemList))

    assert len(items) == 3
    assert items[0].name == "item1"
    assert items[1].name == "item2"
    assert items[2].name == "item3"
