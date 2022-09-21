from attrs import fields

from glotaran.model_new.item import ModelItem
from glotaran.model_new.item import ModelItemType
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import item
from glotaran.model_new.item import model_attributes
from glotaran.model_new.item import strip_type_and_structure_from_attribute
from glotaran.model_new.megacomplex import Megacomplex
from glotaran.model_new.megacomplex import megacomplex
from glotaran.model_new.model import Model
from glotaran.parameter import ParameterGroup


@item
class MockModelItem(ModelItem):
    p_scalar: ParameterType
    p_list: list[ParameterType]
    p_dict: dict[str, ParameterType]


@megacomplex()
class MockMegacomplexItems(Megacomplex):
    type: str = "test_model_items_megacomplex"
    item1: ModelItemType[MockModelItem]
    item2: list[ModelItemType[MockModelItem]]
    item3: dict[str, ModelItemType[MockModelItem]]


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
    items = list(model_attributes(MockMegacomplexItems))

    assert len(items) == 3
    assert items[0].name == "item1"
    assert items[1].name == "item2"
    assert items[2].name == "item3"


def test_get_issues():
    mcls = Model.create_class_from_megacomplexes([MockMegacomplexItems])
    model = mcls(
        megacomplex={
            "m1": {
                "type": "test_model_items_megacomplex",
                "item1": "item1",
                "item2": ["item2"],
                "item3": {"foo": "item3"},
            }
        },
        item1={"test": {"p_scalar": "p1", "p_list": ["p2"], "p_dict": {"p": "p2"}}},
    )

    m = model.megacomplex["m1"]
    issues = m.get_model_issues(model)
    assert len(issues) == 3

    p = ParameterGroup()
    i = model.item1["test"]
    issues = i.get_parameter_issues(p)
    assert len(issues) == 3

    issues = model.get_issues(parameters=p)
    assert len(issues) == 6
