from typing import Dict
from typing import List
from typing import Tuple

import pytest
import xarray as xr
from IPython.core.formatters import format_display_data

from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import megacomplex
from glotaran.model import model_item
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.model import Model
from glotaran.model.relation import Relation
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup


@model_item(
    properties={
        "param": Parameter,
        "megacomplex": str,
        "param_list": List[Parameter],
        "default_item": {"type": int, "default": 42},
        "complex": {"type": Dict[Tuple[str, str], Parameter]},
    },
)
class MockItem:
    pass


@model_item(has_label=False)
class MockItemNoLabel:
    pass


@megacomplex(dimension="model", model_items={"test_item1": {"type": MockItem, "allow_none": True}})
class MockMegacomplex1(Megacomplex):
    pass


@megacomplex(dimension="model", model_items={"test_item2": MockItemNoLabel})
class MockMegacomplex2(Megacomplex):
    pass


@megacomplex(dimension="model", model_items={"test_item3": List[MockItem]})
class MockMegacomplex3(Megacomplex):
    pass


@megacomplex(dimension="model", model_items={"test_item4": Dict[str, MockItem]})
class MockMegacomplex4(Megacomplex):
    pass


@megacomplex(
    dimension="model",
    dataset_model_items={"test_item_dataset": MockItem},
    dataset_properties={
        "test_property_dataset1": int,
        "test_property_dataset2": {"type": Parameter},
    },
)
class MockMegacomplex5(Megacomplex):
    pass


@pytest.fixture
def test_model():
    model_dict = {
        "megacomplex": {
            "m1": {"test_item1": "t2"},
            "m2": {"type": "type5", "dimension": "model2"},
        },
        "weights": [
            {
                "datasets": ["d1", "d2"],
                "global_interval": (1, 4),
                "model_interval": (2, 3),
                "value": 5.4,
            }
        ],
        "test_item1": {
            "t1": {
                "param": "foo",
                "megacomplex": "m1",
                "param_list": ["bar", "baz"],
                "complex": {("s1", "s2"): "baz"},
            },
            "t2": {
                "param": "baz",
                "megacomplex": "m2",
                "param_list": ["foo"],
                "complex": {},
                "default_item": 7,
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
                "scale": "scale_2",
                "test_item_dataset": "t2",
                "test_property_dataset1": 1,
                "test_property_dataset2": "bar",
            },
        },
    }
    model_dict["test_item_dataset"] = model_dict["test_item1"]
    return Model.from_dict(
        model_dict,
        megacomplex_types={
            "type1": MockMegacomplex1,
            "type5": MockMegacomplex5,
        },
    )


@pytest.fixture
def model_error():
    model_dict = {
        "megacomplex": {"m1": {}, "m2": {}},
        "test_item1": {
            "t1": {
                "param": "fool",
                "megacomplex": "mX",
                "param_list": ["bar", "bay"],
                "complex": {("s1", "s3"): "boz"},
            },
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["N1", "N2"],
                "scale": "scale_1",
            },
            "dataset2": {
                "megacomplex": ["mrX"],
                "scale": "scale_3",
            },
        },
    }
    return Model.from_dict(
        model_dict,
        megacomplex_types={
            "type1": MockMegacomplex1,
        },
    )


def test_model_init():
    model = Model(
        megacomplex_types={
            "type1": MockMegacomplex1,
            "type2": MockMegacomplex2,
            "type3": MockMegacomplex3,
            "type4": MockMegacomplex4,
            "type5": MockMegacomplex5,
        }
    )

    assert model.default_megacomplex == "type1"

    assert len(model.megacomplex_types) == 5
    assert "type1" in model.megacomplex_types
    assert model.megacomplex_types["type1"] == MockMegacomplex1
    assert "type2" in model.megacomplex_types
    assert model.megacomplex_types["type2"] == MockMegacomplex2

    assert hasattr(model, "test_item1")
    assert isinstance(model.test_item1, dict)
    assert "test_item1" in model._model_items
    assert issubclass(model._model_items["test_item1"], MockItem)

    assert hasattr(model, "test_item2")
    assert isinstance(model.test_item2, list)
    assert "test_item2" in model._model_items
    assert issubclass(model._model_items["test_item2"], MockItemNoLabel)

    assert hasattr(model, "test_item3")
    assert isinstance(model.test_item3, dict)
    assert "test_item3" in model._model_items
    assert issubclass(model._model_items["test_item3"], MockItem)

    assert hasattr(model, "test_item4")
    assert isinstance(model.test_item4, dict)
    assert "test_item4" in model._model_items
    assert issubclass(model._model_items["test_item4"], MockItem)

    assert hasattr(model, "test_item_dataset")
    assert isinstance(model.test_item_dataset, dict)
    assert "test_item_dataset" in model._model_items
    assert issubclass(model._model_items["test_item_dataset"], MockItem)
    assert "test_item_dataset" in model._dataset_properties
    assert issubclass(model._dataset_properties["test_item_dataset"]["type"], str)
    assert "test_property_dataset1" in model._dataset_properties
    assert issubclass(model._dataset_properties["test_property_dataset1"], int)
    assert "test_property_dataset2" in model._dataset_properties
    assert issubclass(model._dataset_properties["test_property_dataset2"]["type"], Parameter)

    assert hasattr(model, "clp_area_penalties")
    assert isinstance(model.clp_area_penalties, list)
    assert "clp_area_penalties" in model._model_items
    assert issubclass(model._model_items["clp_area_penalties"], EqualAreaPenalty)

    assert hasattr(model, "constraints")
    assert isinstance(model.constraints, list)
    assert "constraints" in model._model_items
    assert issubclass(model._model_items["constraints"], Constraint)

    assert hasattr(model, "relations")
    assert isinstance(model.relations, list)
    assert "relations" in model._model_items
    assert issubclass(model._model_items["relations"], Relation)

    assert hasattr(model, "weights")
    assert isinstance(model.weights, list)
    assert "weights" in model._model_items
    assert issubclass(model._model_items["weights"], Weight)

    assert hasattr(model, "dataset")
    assert isinstance(model.dataset, dict)
    assert "dataset" in model._model_items
    assert issubclass(model._model_items["dataset"], DatasetModel)


@pytest.fixture
def parameter():
    params = [1, 2, ["foo", 3], ["bar", 4], ["baz", 2], ["scale_1", 2], ["scale_2", 8], 4e2]
    return ParameterGroup.from_list(params)


def test_model_misc(test_model: Model):
    assert isinstance(test_model.megacomplex["m1"], MockMegacomplex1)
    assert isinstance(test_model.megacomplex["m2"], MockMegacomplex5)
    assert test_model.megacomplex["m1"].dimension == "model"
    assert test_model.megacomplex["m2"].dimension == "model2"


def test_model_validity(test_model: Model, model_error: Model, parameter: ParameterGroup):
    print(test_model.test_item1["t1"])
    print(test_model.problem_list())
    print(test_model.problem_list(parameter))
    assert test_model.valid()
    assert test_model.valid(parameter)
    print(model_error.problem_list())
    print(model_error.problem_list(parameter))
    assert not model_error.valid()
    assert len(model_error.problem_list()) == 4
    assert not model_error.valid(parameter)
    assert len(model_error.problem_list(parameter)) == 8


def test_items(test_model: Model):

    assert "m1" in test_model.megacomplex
    assert "m2" in test_model.megacomplex

    assert "t1" in test_model.test_item1
    t = test_model.test_item1.get("t1")
    assert t.param.full_label == "foo"
    assert t.megacomplex == "m1"
    assert [p.full_label for p in t.param_list] == ["bar", "baz"]
    assert t.default_item == 42
    assert ("s1", "s2") in t.complex
    assert t.complex[("s1", "s2")].full_label == "baz"
    assert "t2" in test_model.test_item1
    t = test_model.test_item1.get("t2")
    assert t.param.full_label == "baz"
    assert t.megacomplex == "m2"
    assert [p.full_label for p in t.param_list] == ["foo"]
    assert t.default_item == 7
    assert t.complex == {}

    assert "dataset1" in test_model.dataset
    assert test_model.dataset.get("dataset1").megacomplex == ["m1"]
    assert test_model.dataset.get("dataset1").scale.full_label == "scale_1"

    assert "dataset2" in test_model.dataset
    assert test_model.dataset.get("dataset2").megacomplex == ["m2"]
    assert test_model.dataset.get("dataset2").scale.full_label == "scale_2"

    assert len(test_model.weights) == 1
    w = test_model.weights[0]
    assert w.datasets == ["d1", "d2"]
    assert w.global_interval == (1, 4)
    assert w.model_interval == (2, 3)
    assert w.value == 5.4


def test_fill(test_model: Model, parameter: ParameterGroup):
    data = xr.DataArray([[1]], dims=("global", "model")).to_dataset(name="data")
    dataset = test_model.dataset.get("dataset1").fill(test_model, parameter)
    dataset.set_data(data)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ["m1"]
    assert dataset.scale == 2
    assert dataset.get_model_dimension() == "model"
    assert dataset.get_global_dimension() == "global"

    data = xr.DataArray([[1]], dims=("global2", "model2")).to_dataset(name="data")
    dataset = test_model.dataset.get("dataset2").fill(test_model, parameter)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ["m2"]
    assert dataset.scale == 8
    dataset.set_data(data)
    assert dataset.get_model_dimension() == "model2"
    assert dataset.get_global_dimension() == "global2"
    #
    t = test_model.test_item1.get("t1").fill(test_model, parameter)
    assert t.param == 3
    assert t.megacomplex.label == "m1"
    assert t.param_list == [4, 2]
    assert t.default_item == 42
    assert t.complex == {("s1", "s2"): 2}
    t = test_model.test_item1.get("t2").fill(test_model, parameter)
    assert t.param == 2
    assert t.megacomplex.label == "m2"
    assert t.param_list == [3]
    assert t.default_item == 7
    assert t.complex == {}


def test_model_markdown_base_heading_level(test_model: Model):
    """base_heading_level applies to all sections."""
    assert test_model.markdown().startswith("# Model")
    assert "## Test" in test_model.markdown()
    assert test_model.markdown(base_heading_level=3).startswith("### Model")
    assert "#### Test" in test_model.markdown(base_heading_level=3)


def test_model_ipython_rendering(test_model: Model):
    """Autorendering in ipython"""
    rendered_obj = format_display_data(test_model)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("### Model")

    rendered_markdown_return = format_display_data(test_model.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("# Model")
