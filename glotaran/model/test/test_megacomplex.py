from typing import Literal

import numpy as np

from glotaran.model.data_model import DataModel
from glotaran.model.item import LibraryItemType
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.test.test_item import MockLibraryItemNested
from glotaran.model.test.test_item import MockTypedItem


class MockDataModel(DataModel):
    item: LibraryItemType[MockTypedItem]


class MockMegacomplexWithDataModel(Megacomplex):
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


class MockMegacomplexWithItem(Megacomplex):
    type: Literal["mock-w-item"]
    dimension: str = "model"
    library_item: LibraryItemType[MockLibraryItemNested] | None = None

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):

        return ["a"], np.array([[1]])
