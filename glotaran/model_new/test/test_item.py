from glotaran.model_new.item import ModelItemType
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import has_label
from glotaran.model_new.item import model_item


@model_item
class MockModelItemDict:
    label: str
    a_string: str
    p_scalar: ParameterType
    p_scalar_option: ParameterType | None


@model_item
class MockModelItemList:
    item: ModelItemType[MockModelItemDict]


def test_item():
    item_dict = MockModelItemDict(label="item", a_string="X", p_scalar="p_scalar")
    assert has_label(item_dict)
    item_list = MockModelItemList(item="item")
    assert not has_label(item_list)
