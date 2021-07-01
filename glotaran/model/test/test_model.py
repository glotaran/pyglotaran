from typing import Dict
from typing import List
from typing import Tuple

#  from glotaran.model import model
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

#  import pytest
#  import xarray as xr
#  from IPython.core.formatters import format_display_data


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


@megacomplex(dimension="model", model_items={"test_item1": MockItem})
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


#
#  @model(
#      "mock_model",
#      attributes={"test": MockAttr},
#      megacomplex_types={
#          "mock_megacomplex": MockMegacomplex,
#          "mock_megacomplex2": MockMegacomplex2,
#      },
#      model_dimension="model",
#      global_dimension="global",
#  )
#  class MockModel(Model):
#      pass
#
#
#  @pytest.fixture
#  def mock_model():
#      d = {
#          "megacomplex": {"m1": {}, "m2": ["mock_megacomplex2", "model2"]},
#          "weights": [
#              {
#                  "datasets": ["d1", "d2"],
#                  "global_interval": (1, 4),
#                  "model_interval": (2, 3),
#                  "value": 5.4,
#              }
#          ],
#          "test": {
#              "t1": {
#                  "param": "foo",
#                  "megacomplex": "m1",
#                  "param_list": ["bar", "baz"],
#                  "complex": {("s1", "s2"): "baz"},
#              },
#              "t2": ["baz", "m2", ["foo"], 7, {}],
#          },
#          "dataset": {
#              "dataset1": {
#                  "megacomplex": ["m1"],
#                  "scale": "scale_1",
#              },
#              "dataset2": [["m2"], ["bar"], "scale_2"],
#          },
#      }
#      return MockModel.from_dict(d)
#
#
#  @pytest.fixture
#  def model_error():
#      d = {
#          "megacomplex": {"m1": {}, "m2": {}},
#          "test": {
#              "t1": {
#                  "param": "fool",
#                  "megacomplex": "mX",
#                  "param_list": ["bar", "bay"],
#                  "complex": {("s1", "s3"): "boz"},
#              },
#          },
#          "dataset": {
#              "dataset1": {
#                  "megacomplex": ["N1", "N2"],
#                  "scale": "scale_1",
#              },
#              "dataset2": [["mrX"], ["bar"], "scale_3"],
#          },
#      }
#      return MockModel.from_dict(d)
#
#
#  @pytest.fixture
#  def parameter():
#      params = [1, 2, ["foo", 3], ["bar", 4], ["baz", 2], ["scale_1", 2], ["scale_2", 8], 4e2]
#      return ParameterGroup.from_list(params)
#
#
#  def test_model_misc(mock_model: Model):
#      assert mock_model.model_type == "mock_model"
#      assert isinstance(mock_model.megacomplex["m1"], MockMegacomplex)
#      assert isinstance(mock_model.megacomplex["m2"], MockMegacomplex2)
#      assert mock_model.megacomplex["m1"].dimension == "model"
#      assert mock_model.megacomplex["m2"].dimension == "model2"
#
#
#  @pytest.mark.parametrize("attr", ["dataset", "megacomplex", "weights", "test"])
#  def test_model_attr(mock_model: Model, attr: str):
#      assert hasattr(mock_model, attr)
#      if attr != "weights":
#          assert hasattr(mock_model, f"get_{attr}")
#          assert hasattr(mock_model, f"set_{attr}")
#      else:
#          assert hasattr(mock_model, f"add_{attr}")
#
#
#  def test_model_validity(mock_model: Model, model_error: Model, parameter: ParameterGroup):
#      print(mock_model.test["t1"])
#      print(mock_model.problem_list())
#      print(mock_model.problem_list(parameter))
#      assert mock_model.valid()
#      assert mock_model.valid(parameter)
#      print(model_error.problem_list())
#      print(model_error.problem_list(parameter))
#      assert not model_error.valid()
#      assert len(model_error.problem_list()) == 4
#      assert not model_error.valid(parameter)
#      assert len(model_error.problem_list(parameter)) == 8
#
#
#  def test_items(mock_model: Model):
#
#      assert "m1" in mock_model.megacomplex
#      assert "m2" in mock_model.megacomplex
#
#      assert "t1" in mock_model.test
#      t = mock_model.get_test("t1")
#      assert t.param.full_label == "foo"
#      assert t.megacomplex == "m1"
#      assert [p.full_label for p in t.param_list] == ["bar", "baz"]
#      assert t.default_item == 42
#      assert ("s1", "s2") in t.complex
#      assert t.complex[("s1", "s2")].full_label == "baz"
#      assert "t2" in mock_model.test
#      t = mock_model.get_test("t2")
#      assert t.param.full_label == "baz"
#      assert t.megacomplex == "m2"
#      assert [p.full_label for p in t.param_list] == ["foo"]
#      assert t.default_item == 7
#      assert t.complex == {}
#
#      assert "dataset1" in mock_model.dataset
#      assert mock_model.get_dataset("dataset1").megacomplex == ["m1"]
#      assert mock_model.get_dataset("dataset1").scale.full_label == "scale_1"
#
#      assert "dataset2" in mock_model.dataset
#      assert mock_model.get_dataset("dataset2").megacomplex == ["m2"]
#      assert mock_model.get_dataset("dataset2").scale.full_label == "scale_2"
#
#      assert len(mock_model.weights) == 1
#      w = mock_model.weights[0]
#      assert w.datasets == ["d1", "d2"]
#      assert w.global_interval == (1, 4)
#      assert w.model_interval == (2, 3)
#      assert w.value == 5.4
#
#
#  def test_fill(mock_model: Model, parameter: ParameterGroup):
#      data = xr.DataArray([[1]], dims=("global", "model")).to_dataset(name="data")
#      dataset = mock_model.get_dataset("dataset1").fill(mock_model, parameter)
#      dataset.set_data(data)
#      assert [cmplx.label for cmplx in dataset.megacomplex] == ["m1"]
#      assert dataset.scale == 2
#      assert dataset.get_model_dimension() == "model"
#      assert dataset.get_global_dimension() == "global"
#
#      data = xr.DataArray([[1]], dims=("global2", "model2")).to_dataset(name="data")
#      dataset = mock_model.get_dataset("dataset2").fill(mock_model, parameter)
#      assert [cmplx.label for cmplx in dataset.megacomplex] == ["m2"]
#      assert dataset.scale == 8
#      dataset.set_data(data)
#      assert dataset.get_model_dimension() == "model2"
#      assert dataset.get_global_dimension() == "global2"
#
#      t = mock_model.get_test("t1").fill(mock_model, parameter)
#      assert t.param == 3
#      assert t.megacomplex.label == "m1"
#      assert t.param_list == [4, 2]
#      assert t.default_item == 42
#      assert t.complex == {("s1", "s2"): 2}
#      t = mock_model.get_test("t2").fill(mock_model, parameter)
#      assert t.param == 2
#      assert t.megacomplex.label == "m2"
#      assert t.param_list == [3]
#      assert t.default_item == 7
#      assert t.complex == {}
#
#
#  def test_model_markdown_base_heading_level(mock_model: Model):
#      """base_heading_level applies to all sections."""
#      assert mock_model.markdown().startswith("# Model")
#      assert "## Test" in mock_model.markdown()
#      assert mock_model.markdown(base_heading_level=3).startswith("### Model")
#      assert "#### Test" in mock_model.markdown(base_heading_level=3)
#
#
#  def test_model_ipython_rendering(mock_model: Model):
#      """Autorendering in ipython"""
#      rendered_obj = format_display_data(mock_model)[0]
#
#      assert "text/markdown" in rendered_obj
#      assert rendered_obj["text/markdown"].startswith("### Model")
#
#      rendered_markdown_return = format_display_data(mock_model.markdown())[0]
#
#      assert "text/markdown" in rendered_markdown_return
#      assert rendered_markdown_return["text/markdown"].startswith("# Model")
