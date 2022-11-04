from attrs import fields

from glotaran.model.item import ModelItem
from glotaran.model.item import ModelItemType
from glotaran.model.item import ParameterType
from glotaran.model.item import fill_item
from glotaran.model.item import get_item_model_issues
from glotaran.model.item import get_item_parameter_issues
from glotaran.model.item import item
from glotaran.model.item import model_attributes
from glotaran.model.item import strip_type_and_structure_from_attribute
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


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
        structure, type_ = strip_type_and_structure_from_attribute(attr)
        print(attr.name, attr.type, structure, type_)
        assert structure in (None, dict, list)
        assert type_ is int


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
    issues = get_item_model_issues(m, model)
    assert len(issues) == 3

    p = Parameters({})
    i = model.item1["test"]
    issues = get_item_parameter_issues(i, p)
    assert len(issues) == 3

    issues = model.get_issues(parameters=p)
    assert len(issues) == 6


def test_fill_item():
    mcls = Model.create_class_from_megacomplexes([MockMegacomplexItems])
    model = mcls(
        megacomplex={
            "m1": {
                "type": "test_model_items_megacomplex",
                "item1": "item",
                "item2": ["item"],
                "item3": {"foo": "item"},
            }
        },
        item1={"item": {"p_scalar": "1", "p_list": ["2"], "p_dict": {"p": "2"}}},
        item2={"item": {"p_scalar": "1", "p_list": ["2"], "p_dict": {"p": "2"}}},
        item3={"item": {"p_scalar": "1", "p_list": ["2"], "p_dict": {"p": "2"}}},
    )

    parameters = Parameters.from_list([2, 3, 4])
    assert model.valid(parameters)

    m = fill_item(model.megacomplex["m1"], model, parameters)
    assert isinstance(m.item1, MockModelItem)
    assert all(isinstance(v, MockModelItem) for v in m.item2)
    assert all(isinstance(v, MockModelItem) for v in m.item3.values())

    i = m.item1
    assert isinstance(i.p_scalar, Parameter)
    assert all(isinstance(v, Parameter) for v in i.p_list)
    assert all(isinstance(v, Parameter) for v in i.p_dict.values())
    assert i.p_scalar.value == 2
    assert i.p_list[0].value == 3
    assert i.p_dict["p"].value == 3
