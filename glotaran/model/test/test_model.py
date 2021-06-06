from typing import Dict
from typing import List
from typing import Tuple

import pytest
from IPython.core.formatters import format_display_data

from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import model
from glotaran.model import model_attribute
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup


@model_attribute(
    properties={
        "param": Parameter,
        "megacomplex": str,
        "param_list": List[Parameter],
        "default_item": {"type": int, "default": 42},
        "complex": {"type": Dict[Tuple[str, str], Parameter]},
    },
)
class MockAttr:
    pass


@model_attribute()
class MockMegacomplex(Megacomplex):
    pass


@model_attribute()
class MockMegacomplex2(Megacomplex):
    pass


@model(
    "mock_model",
    attributes={"test": MockAttr},
    megacomplex_types={
        "mock_megacomplex": MockMegacomplex,
        "mock_megacomplex2": MockMegacomplex2,
    },
    model_dimension="model",
    global_dimension="global",
)
class MockModel(Model):
    pass


@pytest.fixture
def mock_model():
    d = {
        "megacomplex": {"m1": {}, "m2": ["mock_megacomplex2"]},
        "weights": [
            {
                "datasets": ["d1", "d2"],
                "global_interval": (1, 4),
                "model_interval": (2, 3),
                "value": 5.4,
            }
        ],
        "test": {
            "t1": {
                "param": "foo",
                "megacomplex": "m1",
                "param_list": ["bar", "baz"],
                "complex": {("s1", "s2"): "baz"},
            },
            "t2": ["baz", "m2", ["foo"], 7, {}],
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1", "m2"],
                "scale": "scale_1",
            },
            "dataset2": [["m2"], ["bar"], "scale_2"],
        },
    }
    return MockModel.from_dict(d)


@pytest.fixture
def model_error():
    d = {
        "megacomplex": {"m1": {}, "m2": {}},
        "test": {
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
            "dataset2": [["mrX"], ["bar"], "scale_3"],
        },
    }
    return MockModel.from_dict(d)


@pytest.fixture
def parameter():
    params = [1, 2, ["foo", 3], ["bar", 4], ["baz", 2], ["scale_1", 2], ["scale_2", 8], 4e2]
    return ParameterGroup.from_list(params)


def test_model_misc(mock_model: Model):
    assert mock_model.model_type == "mock_model"
    assert isinstance(mock_model.megacomplex["m1"], MockMegacomplex)
    assert isinstance(mock_model.megacomplex["m2"], MockMegacomplex2)


@pytest.mark.parametrize("attr", ["dataset", "megacomplex", "weights", "test"])
def test_model_attr(mock_model: Model, attr: str):
    assert hasattr(mock_model, attr)
    if attr != "weights":
        assert hasattr(mock_model, f"get_{attr}")
        assert hasattr(mock_model, f"set_{attr}")
    else:
        assert hasattr(mock_model, f"add_{attr}")


def test_model_validity(mock_model: Model, model_error: Model, parameter: ParameterGroup):
    print(mock_model.test["t1"])
    print(mock_model.problem_list())
    print(mock_model.problem_list(parameter))
    assert mock_model.valid()
    assert mock_model.valid(parameter)
    print(model_error.problem_list())
    print(model_error.problem_list(parameter))
    assert not model_error.valid()
    assert len(model_error.problem_list()) == 4
    assert not model_error.valid(parameter)
    assert len(model_error.problem_list(parameter)) == 8


def test_items(mock_model: Model):

    assert "m1" in mock_model.megacomplex
    assert "m2" in mock_model.megacomplex

    assert "t1" in mock_model.test
    t = mock_model.get_test("t1")
    assert t.param.full_label == "foo"
    assert t.megacomplex == "m1"
    assert [p.full_label for p in t.param_list] == ["bar", "baz"]
    assert t.default_item == 42
    assert ("s1", "s2") in t.complex
    assert t.complex[("s1", "s2")].full_label == "baz"
    assert "t2" in mock_model.test
    t = mock_model.get_test("t2")
    assert t.param.full_label == "baz"
    assert t.megacomplex == "m2"
    assert [p.full_label for p in t.param_list] == ["foo"]
    assert t.default_item == 7
    assert t.complex == {}

    assert "dataset1" in mock_model.dataset
    assert mock_model.get_dataset("dataset1").megacomplex == ["m1", "m2"]
    assert mock_model.get_dataset("dataset1").scale.full_label == "scale_1"

    assert "dataset2" in mock_model.dataset
    assert mock_model.get_dataset("dataset2").megacomplex == ["m2"]
    assert mock_model.get_dataset("dataset2").scale.full_label == "scale_2"

    assert len(mock_model.weights) == 1
    w = mock_model.weights[0]
    assert w.datasets == ["d1", "d2"]
    assert w.global_interval == (1, 4)
    assert w.model_interval == (2, 3)
    assert w.value == 5.4


def test_fill(mock_model: Model, parameter: ParameterGroup):
    dataset = mock_model.get_dataset("dataset1").fill(mock_model, parameter)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ["m1", "m2"]
    assert dataset.scale == 2

    dataset = mock_model.get_dataset("dataset2").fill(mock_model, parameter)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ["m2"]
    assert dataset.scale == 8

    t = mock_model.get_test("t1").fill(mock_model, parameter)
    assert t.param == 3
    assert t.megacomplex.label == "m1"
    assert t.param_list == [4, 2]
    assert t.default_item == 42
    assert t.complex == {("s1", "s2"): 2}
    t = mock_model.get_test("t2").fill(mock_model, parameter)
    assert t.param == 2
    assert t.megacomplex.label == "m2"
    assert t.param_list == [3]
    assert t.default_item == 7
    assert t.complex == {}


def test_model_markdown_base_heading_level(mock_model: Model):
    """base_heading_level applies to all sections."""
    assert mock_model.markdown().startswith("# Model")
    assert "## Test" in mock_model.markdown()
    assert mock_model.markdown(base_heading_level=3).startswith("### Model")
    assert "#### Test" in mock_model.markdown(base_heading_level=3)


def test_model_ipython_rendering(mock_model: Model):
    """Autorendering in ipython"""
    rendered_obj = format_display_data(mock_model)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("### Model")

    rendered_markdown_return = format_display_data(mock_model.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("# Model")
