from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.element import Element
from glotaran.model.item import get_item_issues
from glotaran.parameter import Parameters
from tests.model.test_item import MockItem  # noqa: TCH001
from tests.model.test_item import MockTypedItem  # noqa: TCH001

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class MockDataModel(DataModel):
    item: MockTypedItem | None = None


class MockElementWithDataModel(Element):
    type: Literal["mock-w-datamodel"]
    dimension: str = "model"
    data_model_type: ClassVar[type[DataModel]] = MockDataModel

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        return ["a"], np.array([[1]])

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


class MockElementWithItem(Element):
    type: Literal["mock-w-item"]
    dimension: str = "model"
    item: MockItem | None = None

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        return ["a"], np.array([[1]])

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


class MockElementNonUniqueExclusive(Element):
    type: Literal["test_element_not_exclusive_unique"]

    def calculate_matrix():
        pass

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


class MockElementExclusive(Element):
    type: Literal["test_element_exclusive"]
    is_exclusive = True

    def calculate_matrix():
        pass

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


class MockElementUnique(Element):
    type: Literal["test_element_unique"]
    is_unique = True

    def calculate_matrix():
        pass

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


class MockElementDim1(Element):
    type: Literal["test_element_dim1"]
    dimension: str = "dim1"

    def calculate_matrix():
        pass

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


class MockElementDim2(Element):
    type: Literal["test_element_dim2"]
    dimension: str = "dim2"

    def calculate_matrix():
        pass

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()


def test_data_model_from_dict():
    library = {
        "m1": MockElementWithDataModel(label="m1", type="mock-w-datamodel"),
        "m2": MockElementWithItem(label="m2", type="mock-w-item"),
    }
    d1 = DataModel.from_dict(library, {"elements": ["m1"]})
    assert isinstance(d1, MockDataModel)

    d2 = DataModel.from_dict(library, {"elements": ["m2"]})
    assert isinstance(d2, DataModel)
    assert not isinstance(d2, MockDataModel)

    d3 = DataModel.from_dict(library, {"elements": ["m2"], "global_elements": ["m1"]})
    assert isinstance(d3, MockDataModel)


def test_get_data_model_issues():
    ok = DataModel(
        **{
            "elements": [
                MockElementNonUniqueExclusive(label="m", type="test_element_not_exclusive_unique")
            ]
        },
    )
    exclusive = DataModel(
        **{
            "elements": [
                MockElementNonUniqueExclusive(label="m", type="test_element_not_exclusive_unique"),
                MockElementExclusive(label="m_exclusive", type="test_element_exclusive"),
            ]
        },
    )
    unique = DataModel(
        **{
            "elements": [
                MockElementUnique(label="m_unique", type="test_element_unique"),
                MockElementUnique(label="m_unique", type="test_element_unique"),
            ]
        },
    )

    assert len(get_item_issues(ok, Parameters({}))) == 0
    assert len(get_item_issues(exclusive, Parameters({}))) == 1
    assert len(get_item_issues(unique, Parameters({}))) == 2
