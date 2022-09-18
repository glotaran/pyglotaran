from glotaran.model_new.item import ModelItem
from glotaran.model_new.item import ModelItemType
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import has_label
from glotaran.model_new.item import infer_model_item_type_from_attribute
from glotaran.model_new.item import item
from glotaran.model_new.item import model_item
from glotaran.model_new.item import model_items


@item
class MockModelItemDict(ModelItem):
    a_string: str
    p_scalar: ParameterType
    p_scalar_option: ParameterType | None = None


@item
class MockModelItemList:
    item1: ModelItemType[MockModelItemDict] = model_item()
    item2: list[ModelItemType[MockModelItemDict]] = model_item()
    item3: dict[ModelItemType[MockModelItemDict]] = model_item()


def test_item():
    item_dict = MockModelItemDict(label="item", a_string="X", p_scalar="p_scalar")
    assert has_label(item_dict)
    item_list = MockModelItemList(
        item1="item",
        item2="item",
        item3="item",
    )
    assert not has_label(item_list)


def test_model_items():
    items = list(model_items(MockModelItemList))

    assert len(items) == 3
    assert items[0].name == "item1"
    assert items[1].name == "item2"
    assert items[2].name == "item3"


def test_infer_model_item_type():
    @item
    class MockModelItem(ModelItem):
        scalar: ModelItemType[int] = model_item(None)
        scalar_option: ModelItemType[int] | None = model_item(None)
        ilist: list[ModelItemType[int]] = model_item(None)
        ilist_option: list[ModelItemType[int]] | None = model_item(None)
        idict: dict[str, ModelItemType[int]] = model_item(None)
        idict_option: dict[str, ModelItemType[int]] | None = model_item(None)

    attributes = list(model_items(MockModelItem))
    for i in range(6):
        assert infer_model_item_type_from_attribute(attributes[i]) is int
