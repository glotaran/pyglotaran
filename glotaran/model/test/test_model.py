from math import inf
from math import nan
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
from glotaran.model.constraint import OnlyConstraint
from glotaran.model.constraint import ZeroConstraint
from glotaran.model.interval_property import IntervalProperty
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


@model_item(
    properties={
        "param": Parameter,
        "param_list": List[Parameter],
        "param_dict": {"type": Dict[Tuple[str, str], Parameter]},
        "number": int,
    },
)
class MockItemSimple:
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


@megacomplex(model_items={"test_item3": List[MockItem]})
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


@megacomplex(dimension="model", unique=True)
class MockMegacomplex6(Megacomplex):
    pass


@megacomplex(dimension="model", model_items={"test_item_simple": MockItemSimple})
class MockMegacomplex7(Megacomplex):
    pass


@pytest.fixture
def test_model_dict():
    model_dict = {
        "megacomplex": {
            "m1": {"test_item1": "t2"},
            "m2": {"type": "type5", "dimension": "model2"},
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
                "global_megacomplex": ["m1"],
                "scale": "scale_2",
                "test_item_dataset": "t2",
                "test_property_dataset1": 1,
                "test_property_dataset2": "bar",
                "group": "testgroup",
            },
        },
    }
    model_dict["test_item_dataset"] = model_dict["test_item1"]
    return model_dict


@pytest.fixture
def test_model(test_model_dict):
    return Model.from_dict(
        test_model_dict,
        megacomplex_types={
            "type1": MockMegacomplex1,
            "type5": MockMegacomplex5,
        },
    )


@pytest.fixture
def model_error():
    model_dict = {
        "megacomplex": {"m1": {}, "m2": {"type": "type2"}, "m3": {"type": "type2"}},
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
            "dataset3": {
                "megacomplex": ["m2", "m3"],
            },
        },
    }
    return Model.from_dict(
        model_dict,
        megacomplex_types={
            "type1": MockMegacomplex1,
            "type2": MockMegacomplex6,
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

    assert hasattr(model, "clp_constraints")
    assert isinstance(model.clp_constraints, list)
    assert "clp_constraints" in model._model_items
    assert issubclass(model._model_items["clp_constraints"], Constraint)

    assert hasattr(model, "clp_relations")
    assert isinstance(model.clp_relations, list)
    assert "clp_relations" in model._model_items
    assert issubclass(model._model_items["clp_relations"], Relation)

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


def test_dataset_group_models(test_model: Model):
    groups = test_model.dataset_group_models
    assert "default" in groups
    assert groups["default"].residual_function == "variable_projection"
    assert groups["default"].link_clp is None
    assert "testgroup" in groups
    assert groups["testgroup"].residual_function == "non_negative_least_squares"
    assert groups["testgroup"].link_clp


def test_dataset_groups(test_model: Model):
    groups = test_model.get_dataset_groups()
    assert "default" in groups
    assert groups["default"].model.residual_function == "variable_projection"
    assert groups["default"].model.link_clp is None
    assert "dataset1" in groups["default"].dataset_models
    assert "testgroup" in groups
    assert groups["testgroup"].model.residual_function == "non_negative_least_squares"
    assert groups["testgroup"].model.link_clp
    assert "dataset2" in groups["testgroup"].dataset_models


def test_model_validity(test_model: Model, model_error: Model, parameter: ParameterGroup):
    print(test_model.test_item1["t1"])
    print(test_model.problem_list())
    print(test_model.problem_list(parameter))
    assert test_model.valid()
    assert test_model.valid(parameter)
    print(model_error.problem_list())
    print(model_error.problem_list(parameter))
    assert not model_error.valid()
    assert len(model_error.problem_list()) == 5
    assert not model_error.valid(parameter)
    assert len(model_error.problem_list(parameter)) == 9


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
    assert test_model.dataset.get("dataset2").global_megacomplex == ["m1"]
    assert test_model.dataset.get("dataset2").scale.full_label == "scale_2"

    assert len(test_model.weights) == 1
    w = test_model.weights[0]
    assert w.datasets == ["d1", "d2"]
    assert w.global_interval == (1, 4)
    assert w.model_interval == (2, 3)
    assert w.value == 5.4


def test_fill(test_model: Model, parameter: ParameterGroup):
    data = xr.DataArray([[1]], coords=(("global", [0]), ("model", [0]))).to_dataset(name="data")
    dataset = test_model.dataset.get("dataset1").fill(test_model, parameter)
    dataset.set_data(data)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ["m1"]
    assert dataset.scale == 2

    assert dataset.get_model_dimension() == "model"
    assert dataset.get_global_dimension() == "global"
    dataset.swap_dimensions()
    assert dataset.get_model_dimension() == "global"
    assert dataset.get_global_dimension() == "model"
    dataset.swap_dimensions()
    assert dataset.get_model_dimension() == "model"
    assert dataset.get_global_dimension() == "global"

    assert not dataset.has_global_model()

    dataset = test_model.dataset.get("dataset2").fill(test_model, parameter)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ["m2"]
    assert dataset.scale == 8
    assert dataset.get_model_dimension() == "model2"
    assert dataset.get_global_dimension() == "model"

    assert dataset.has_global_model()
    assert [cmplx.label for cmplx in dataset.global_megacomplex] == ["m1"]

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


def test_model_as_dict():
    model_dict = {
        "default_megacomplex": "type7",
        "megacomplex": {
            "m1": {"test_item_simple": "t2", "dimension": "model"},
        },
        "test_item_simple": {
            "t1": {
                "param": "foo",
                "param_list": ["bar", "baz"],
                "param_dict": {("s1", "s2"): "baz"},
                "number": 21,
            },
        },
        "dataset_groups": {
            "default": {"link_clp": None, "residual_function": "variable_projection"}
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
                "scale": "scale_1",
                "group": "default",
            },
        },
    }
    model = Model.from_dict(
        model_dict,
        megacomplex_types={
            "type7": MockMegacomplex7,
        },
    )
    as_model_dict = model.as_dict()
    assert as_model_dict == model_dict


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


def test_interval_property():
    ip1 = IntervalProperty.from_dict({"interval": [[1, 1000]]})
    assert all(ip1.applies(x) for x in (1, 500, 100))
    assert all(not ip1.applies(x) for x in (9999, inf, nan))


def test_zero_constraint():
    zc1 = ZeroConstraint.from_dict({"interval": [[1, 400], [600, 1000]], "target": "s1"})
    assert all(zc1.applies(x) for x in (1, 2, 400, 600, 1000))
    assert all(not zc1.applies(x) for x in (400.01, 500, 599.99, 9999, inf, nan))
    assert zc1.target == "s1"
    zc2 = ZeroConstraint.from_dict({"interval": [[600, 700]], "target": "s2"})
    assert all(zc2.applies(x) for x in range(600, 700, 50))
    assert all(not zc2.applies(x) for x in (599.9999, 700.0001))
    assert zc2.target == "s2"


def test_only_constraint():
    oc1 = OnlyConstraint.from_dict({"interval": [[1, 400], (600, 1000)], "target": "spectra1"})
    assert all(oc1.applies(x) for x in (400.01, 500, 599.99, 9999, inf))
    assert all(not oc1.applies(x) for x in (1, 400, 600, 1000))
    assert oc1.target == "spectra1"
    oc2 = OnlyConstraint.from_dict({"interval": [(600, 700)], "target": "spectra2"})
    assert oc2.applies(599)
    assert not oc2.applies(650)
    assert oc2.applies(701)
    assert oc2.target == "spectra2"
