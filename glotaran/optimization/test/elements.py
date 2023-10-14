from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model import DataModel
from glotaran.model import Element
from glotaran.model import ParameterType

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class TestElementConstant(Element):
    type: Literal["test-element-constant"]
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

    def add_to_result_data(
        self,
        model: DataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        data.attrs["custom_element_result"] = True


class TestElementExponential(Element):
    type: Literal["test-element-exponential"]
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


class TestElementGaussian(Element):
    type: Literal["test-element-gaussian"]
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
