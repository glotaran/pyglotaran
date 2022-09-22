import pytest

from glotaran.model_new.dataset_model import DatasetModel
from glotaran.model_new.item import ModelItem
from glotaran.model_new.item import ModelItemType
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import item
from glotaran.model_new.megacomplex import Megacomplex
from glotaran.model_new.megacomplex import megacomplex
from glotaran.model_new.model import DEFAULT_DATASET_GROUP
from glotaran.model_new.model import Model


@item
class MockItemSimple(ModelItem):
    param: ParameterType
    param_list: list[ParameterType]
    param_dict: dict[tuple[str, str], ParameterType]
    megacomplex: ModelItemType[Megacomplex]
    number: int = 42


@megacomplex()
class MockMegacomplexSimple(Megacomplex):
    type: str = "simple"
    dimension: str = "model"
    test_item: ModelItemType[MockItemSimple] | None


@megacomplex()
class MockMegacomplexItemList(Megacomplex):
    type: str = "list"
    dimension: str = "model"
    test_item_in_list: list[ModelItemType[MockItemSimple]]


@megacomplex()
class MockMegacomplexItemDict(Megacomplex):
    type: str = "dict"
    dimension: str = "model"
    test_item_in_dict: dict[str, ModelItemType[MockItemSimple]]


@item
class MockDatasetModel(DatasetModel):
    test_item_dataset: ModelItemType[MockItemSimple]
    test_property_dataset1: int
    test_property_dataset2: ParameterType


@megacomplex(dataset_model_type=MockDatasetModel)
class MockMegacomplexWithDataset(Megacomplex):
    type: str = "dataset"
    dimension: str = "model"


@megacomplex(unique=True)
class MockMegacomplexUnique(Megacomplex):
    type: str = "unique"
    dimension: str = "model"


@megacomplex(exclusive=True)
class MockMegacomplexExclusive(Megacomplex):
    type: str = "exclusive"
    dimension: str = "model"


@pytest.fixture
def test_model_dict():
    model_dict = {
        "megacomplex": {
            "m1": {"type": "simple", "test_item": "t2"},
            "m2": {"type": "dataset", "dimension": "model2"},
        },
        "dataset_groups": {
            "testgroup": {"residual_function": "non_negative_least_squares", "link_clp": True}
        },
        "weights": [
            {
                "datasets": ["d1", "d2"],
                "global_interval": (1, 4),
                "model_interval": (2, 3),
                "value": 5.4,
            }
        ],
        "test_item": {
            "t1": {
                "param": "foo",
                "megacomplex": "m1",
                "param_list": ["bar", "baz"],
                "param_dict": {("s1", "s2"): "baz"},
            },
            "t2": {
                "param": "baz",
                "megacomplex": "m2",
                "param_list": ["foo"],
                "param_dict": {},
                "number": 7,
            },
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
                "scale": "scale_1",
                "test_item_dataset": "t1",
                "test_property_dataset1": 1,
                "test_property_dataset2": "bar",
            },
            "dataset2": {
                "megacomplex": ["m2"],
                "global_megacomplex": ["m1"],
                "scale": "scale_2",
                "test_item_dataset": "t2",
                "test_property_dataset1": 1,
                "test_property_dataset2": "bar",
                "group": "testgroup",
            },
        },
    }
    model_dict["test_item_dataset"] = model_dict["test_item"]
    return model_dict


@pytest.fixture
def test_model(test_model_dict):
    mcls = Model.create_class_from_megacomplexes(
        [MockMegacomplexSimple, MockMegacomplexWithDataset]
    )
    return mcls(**test_model_dict)


def test_model_create_class():
    m = Model.create_class([])(dataset={})
    print(m)
    assert DEFAULT_DATASET_GROUP in m.dataset_groups

    m = Model.create_class([])(
        **{
            "dataset": {},
            "dataset_groups": {
                "test": {"residual_function": "non_negative_least_squares", "link_clp": False}
            },
        }
    )
    print(m)
    assert DEFAULT_DATASET_GROUP in m.dataset_groups
    assert "test" in m.dataset_groups
    assert m.dataset_groups["test"].residual_function == "non_negative_least_squares"
    assert not m.dataset_groups["test"].link_clp


def test_global_items():

    m = Model.create_class([])(
        **{
            "dataset": {},
            "weights": [
                {"datasets": ["d1", "d2"], "value": 1},
                {"datasets": ["d3"], "value": 2, "global_interval": (5, 6)},
            ],
        }
    )
    print(m)
    assert len(m.weights) == 2
    w = m.weights[0]
    assert w.datasets == ["d1", "d2"]
    assert w.value == 1
    assert w.model_interval is None
    assert w.global_interval is None

    w = m.weights[1]
    assert w.datasets == ["d3"]
    assert w.value == 2
    assert w.model_interval is None
    assert w.global_interval == (5, 6)


def test_model_items(test_model: Model):
    assert isinstance(test_model.megacomplex["m1"], MockMegacomplexSimple)
    assert isinstance(test_model.megacomplex["m2"], MockMegacomplexWithDataset)
    assert test_model.megacomplex["m1"].dimension == "model"
    assert test_model.megacomplex["m2"].dimension == "model2"
    assert test_model.megacomplex["m1"].test_item == "t2"

    assert test_model.test_item["t1"].param == "foo"
    assert test_model.test_item["t1"].param_list == ["bar", "baz"]
    assert test_model.test_item["t1"].param_dict == {("s1", "s2"): "baz"}
    assert test_model.test_item["t1"].megacomplex == "m1"
    assert test_model.test_item["t1"].number == 42
    assert test_model.test_item["t2"].param == "baz"
    assert test_model.test_item["t2"].param_list == ["foo"]
    assert test_model.test_item["t2"].param_dict == {}
    assert test_model.test_item["t2"].megacomplex == "m2"
    assert test_model.test_item["t2"].number == 7

    assert test_model.dataset["dataset1"].megacomplex == ["m1"]
    assert test_model.dataset["dataset1"].global_megacomplex is None
    assert test_model.dataset["dataset1"].scale == "scale_1"
    assert test_model.dataset["dataset1"].test_item_dataset == "t1"
    assert test_model.dataset["dataset1"].test_property_dataset1 == 1
    assert test_model.dataset["dataset1"].test_property_dataset2 == "bar"
    assert test_model.dataset["dataset1"].group == DEFAULT_DATASET_GROUP

    assert test_model.dataset["dataset2"].megacomplex == ["m2"]
    assert test_model.dataset["dataset2"].global_megacomplex == ["m1"]
    assert test_model.dataset["dataset2"].scale == "scale_2"
    assert test_model.dataset["dataset2"].test_item_dataset == "t2"
    assert test_model.dataset["dataset2"].test_property_dataset1 == 1
    assert test_model.dataset["dataset2"].test_property_dataset2 == "bar"
    assert test_model.dataset["dataset2"].group == "testgroup"


def test_model_as_dict():
    model_dict = {
        "megacomplex": {
            "m1": {"type": "simple", "label": "m1", "dimension": "model", "test_item": "t1"},
        },
        "dataset_groups": {
            "default": {
                "label": "default",
                "residual_function": "non_negative_least_squares",
                "link_clp": True,
            }
        },
        "test_item": {
            "t1": {
                "label": "t1",
                "number": 4,
                "param": "foo",
                "megacomplex": "m1",
                "param_list": ["bar", "baz"],
                "param_dict": {("s1", "s2"): "baz"},
            },
        },
        "dataset": {
            "dataset1": {
                "label": "dataset1",
                "group": "default",
                "megacomplex": ["m1"],
                "global_megacomplex": ["m1"],
                "scale": "scale_1",
                "megacomplex_scale": "scale_1",
                "global_megacomplex_scale": "scale_1",
                "force_index_dependent": False,
            },
        },
        "weights": [],
    }
    as_model_dict = Model.create_class_from_megacomplexes([MockMegacomplexSimple])(
        **model_dict
    ).as_dict()
    print("want")
    print(model_dict)
    print("got")
    print(as_model_dict)
    assert as_model_dict == model_dict
