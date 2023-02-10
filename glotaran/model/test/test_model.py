from typing import Literal

import numpy as np

from glotaran.model.data_model import DataModel
from glotaran.model.model import Model
from glotaran.model.test.test_item import MockItem
from glotaran.model.test.test_item import MockTypedItem


class MockDataModel(DataModel):
    item: MockTypedItem | None = None


class MockModelWithDataModel(Model):
    type: Literal["mock-w-datamodel"]
    dimension: str = "model"
    data_model_type = MockDataModel

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):

        return ["a"], np.array([[1]])


class MockModelWithItem(Model):
    type: Literal["mock-w-item"]
    dimension: str = "model"
    item: MockItem | None = None

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):

        return ["a"], np.array([[1]])
