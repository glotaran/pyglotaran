from typing import Literal

from glotaran.model.data_model import DataModel
from glotaran.model.item import get_item_issues
from glotaran.model.model import Model
from glotaran.model.test.test_model import MockDataModel
from glotaran.model.test.test_model import MockModelWithDataModel
from glotaran.model.test.test_model import MockModelWithItem
from glotaran.parameter import Parameters


class MockModelNonUniqueExclusive(Model):
    type: Literal["test_model_not_exclusive_unique"]

    def calculate_matrix():
        pass


class MockModelExclusive(Model):
    type: Literal["test_model_exclusive"]
    is_exclusive = True

    def calculate_matrix():
        pass


class MockModelUnique(Model):
    type: Literal["test_model_unique"]
    is_unique = True

    def calculate_matrix():
        pass


class MockModelDim1(Model):
    type: Literal["test_model_dim1"]
    dimension: str = "dim1"

    def calculate_matrix():
        pass


class MockModelDim2(Model):
    type: Literal["test_model_dim2"]
    dimension: str = "dim2"

    def calculate_matrix():
        pass


def test_data_model_from_dict():
    library = {
        "m1": MockModelWithDataModel(label="m1", type="mock-w-datamodel"),
        "m2": MockModelWithItem(label="m2", type="mock-w-item"),
    }
    d1 = DataModel.from_dict(library, {"models": ["m1"]})
    assert isinstance(d1, MockDataModel)

    d2 = DataModel.from_dict(library, {"models": ["m2"]})
    assert isinstance(d2, DataModel)
    assert not isinstance(d2, MockDataModel)

    d3 = DataModel.from_dict(library, {"models": ["m2"], "global_models": ["m1"]})
    assert isinstance(d3, MockDataModel)


def test_get_data_model_issues():
    ok = DataModel(
        **{
            "models": [
                MockModelNonUniqueExclusive(label="m", type="test_model_not_exclusive_unique")
            ]
        },
    )
    exclusive = DataModel(
        **{
            "models": [
                MockModelNonUniqueExclusive(label="m", type="test_model_not_exclusive_unique"),
                MockModelExclusive(label="m_exclusive", type="test_model_exclusive"),
            ]
        },
    )
    unique = DataModel(
        **{
            "models": [
                MockModelUnique(label="m_unique", type="test_model_unique"),
                MockModelUnique(label="m_unique", type="test_model_unique"),
            ]
        },
    )

    assert len(get_item_issues(ok, Parameters({}))) == 0
    assert len(get_item_issues(exclusive, Parameters({}))) == 1
    assert len(get_item_issues(unique, Parameters({}))) == 2
