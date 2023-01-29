from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model import DataModel
from glotaran.model import Megacomplex
from glotaran.model import ParameterType

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class TestMegacomplexConstant(Megacomplex):
    type: Literal["test-megacomplex-constant"]
    is_index_dependent: bool
    compartments: list[str]
    value: ParameterType

    def calculate_matrix(
        self,
        data_model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ):

        matrix = np.ones((model_axis.size, len(self.compartments))) * float(self.value)
        if self.is_index_dependent:
            matrix = np.array([matrix] * global_axis.size)
        return self.compartments, matrix


class TestMegacomplexExponential(Megacomplex):
    type: Literal["test-megacomplex-exponential"]
    dimension: str = "model"
    is_index_dependent: bool
    compartments: list[str]
    rates: list[ParameterType]

    def calculate_matrix(
        self,
        data_model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ):
        assert len(self.compartments) == len(self.rates)
        rates = -1 * np.asarray(self.rates)
        matrix = np.exp(np.outer(model_axis, rates))
        if self.is_index_dependent:
            matrix = np.array([matrix] * global_axis.size)
        return self.compartments, matrix


class TestMegacomplexGaussian(Megacomplex):
    type: Literal["test-megacomplex-gaussian"]
    compartments: list[str]
    amplitude: list[ParameterType]
    location: list[ParameterType]
    width: list[ParameterType]

    def calculate_matrix(
        self,
        data_model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ):
        amplitude = np.asarray(self.amplitude)
        location = np.asarray(self.location)
        width = np.asarray(self.width)

        matrix = np.empty((model_axis.size, location.size), dtype=np.float64)

        for i in range(location.size):
            matrix[:, i] = amplitude[i] * np.exp(
                -np.log(2) * np.square(2 * (model_axis - location[i]) / width[i])
            )
        return self.compartments, matrix


TestDataModelConstantIndexIndependent = DataModel(
    data=xr.DataArray(
        np.ones((4, 3)) * 5, coords=[("model", [5, 7, 9, 12]), ("global", [1, 5, 6])]
    ).to_dataset(name="data"),
    megacomplex=[
        TestMegacomplexConstant(
            type="test-megacomplex-constant",
            label="test",
            dimension="model",
            compartments=["c5"],
            value=5,
            is_index_dependent=False,
        )
    ],
)

TestDataModelConstantIndexDependent = DataModel(
    data=xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", [0, 3, 7, 10]), ("model", [4, 11, 15])]
    ).to_dataset(name="data"),
    megacomplex=[
        TestMegacomplexConstant(
            type="test-megacomplex-constant",
            label="test",
            dimension="model",
            compartments=["c2"],
            value=2,
            is_index_dependent=True,
        )
    ],
)

TestDataModelGlobal = DataModel(
    data=xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", [0, 3, 7, 10]), ("model", [4, 11, 15])]
    ).to_dataset(name="data"),
    megacomplex=[
        TestMegacomplexConstant(
            type="test-megacomplex-constant",
            label="test",
            dimension="model",
            compartments=["c2"],
            value=5,
            is_index_dependent=False,
        )
    ],
    global_megacomplex=[
        TestMegacomplexConstant(
            type="test-megacomplex-constant",
            label="test_global",
            dimension="global",
            compartments=["c2"],
            value=4,
            is_index_dependent=False,
        )
    ],
)

TestDataModelConstantThreeCompartments = DataModel(
    data=xr.DataArray(
        np.ones((6, 5)) * 3, coords=[("global", [1, 3, 7, 8, 9, 10]), ("model", [4, 5, 7, 11, 15])]
    ).to_dataset(name="data"),
    megacomplex=[
        TestMegacomplexConstant(
            type="test-megacomplex-constant",
            label="test",
            dimension="model",
            compartments=[
                "c3_1",
                "c3_2",
                "c3_3",
            ],
            value=3,
            is_index_dependent=False,
        )
    ],
)
