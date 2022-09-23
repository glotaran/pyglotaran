from glotaran.model.dataset_model import DatasetModel
from glotaran.model.item import ModelItem
from glotaran.model.item import ModelItemType
from glotaran.model.item import item
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model


@item
class MockItem(ModelItem):
    value: int


@item
class MockDatasetModel1(DatasetModel):
    test_dataset_prop: int


@megacomplex(dataset_model_type=MockDatasetModel1)
class MockMegacomplex1(Megacomplex):
    type: str = "mock-complex-1"
    test_item: ModelItemType[MockItem]


@item
class MockDatasetModel2(DatasetModel):
    test_dataset_str: str


@megacomplex(dataset_model_type=MockDatasetModel2)
class MockMegacomplex2(Megacomplex):
    type: str = "mock-complex-2"
    test_str: str = "foo"


def test_add_item_fields_to_model():
    mcls = Model.create_class_from_megacomplexes([MockMegacomplex1])
    m = mcls()
    print(m)
    assert isinstance(m.dataset, dict)
    assert m.dataset == {}
    assert isinstance(m.test_item, dict)
    assert m.test_item == {}

    m = mcls(
        dataset={"d1": {"megacomplex": ["m1"], "test_dataset_prop": 21}},
        megacomplex={"m1": {"type": "mock-complex-1", "test_item": "item1"}},
        test_item={"item1": {"value": 42}},
    )
    print(m)
    assert "m1" in m.megacomplex
    assert isinstance(m.megacomplex["m1"], MockMegacomplex1)
    assert m.megacomplex["m1"].test_item == "item1"

    assert "d1" in m.dataset
    assert m.dataset["d1"].label == "d1"
    assert m.dataset["d1"].megacomplex == ["m1"]
    assert m.dataset["d1"].test_dataset_prop == 21

    assert "item1" in m.test_item
    assert m.test_item["item1"].label == "item1"
    assert m.test_item["item1"].value == 42
