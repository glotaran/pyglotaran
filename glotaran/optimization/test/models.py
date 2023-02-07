from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import ParameterType
from glotaran.model import item
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@megacomplex()
class SimpleTestMegacomplex(Megacomplex):
    type: str = "simple-test-mc"
    dimension: str = "model"
    is_index_dependent: bool

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        compartments = ["s1", "s2"]
        array = np.zeros((model_axis.size, len(compartments)))

        for i in range(len(compartments)):
            for j in range(model_axis.size):
                array[j, i] = (i + j) * model_axis[j]
        if self.is_index_dependent:
            array = np.array([array] * global_axis.size)
        return compartments, array

    def finalize_data(
        self,
        dataset_model,
        dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        # not needed in the tests
        pass


SimpleTestModel = Model.create_class_from_megacomplexes([SimpleTestMegacomplex])


@item
class SimpleDatasetModel(DatasetModel):
    kinetic: list[ParameterType]


@megacomplex(dataset_model_type=SimpleDatasetModel)
class SimpleKineticMegacomplex(Megacomplex):
    type: str = "simple-kinetic-test-mc"
    dimension: str = "model"
    is_index_dependent: bool

    def calculate_matrix(
        self,
        dataset_model,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        kinpar = -1 * np.asarray(dataset_model.kinetic)
        if dataset_model.label == "dataset3":
            # this case is for the ThreeDatasetDecay test
            compartments = [f"s{i+2}" for i in range(len(kinpar))]
        else:
            compartments = [f"s{i+1}" for i in range(len(kinpar))]
        array = np.exp(np.outer(model_axis, kinpar))
        if self.is_index_dependent:
            array = np.array([array] * global_axis.size)
        return compartments, array

    def finalize_data(
        self,
        dataset_model,
        dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


@megacomplex()
class SimpleSpectralMegacomplex(Megacomplex):
    type: str = "simple-spectral-test-mc"
    dimension: str = "global"

    def calculate_matrix(
        self,
        dataset_model,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        kinpar = dataset_model.kinetic
        if dataset_model.label == "dataset3":
            # this case is for the ThreeDatasetDecay test
            compartments = [f"s{i+2}" for i in range(len(kinpar))]
        else:
            compartments = [f"s{i+1}" for i in range(len(kinpar))]
        array = np.asarray([[1 for _ in range(model_axis.size)] for _ in compartments]).T
        return compartments, array


@megacomplex()
class ShapedSpectralMegacomplex(Megacomplex):
    type: str = "shaped-spectral-test-mc"
    dimension: str = "global"
    location: list[ParameterType]
    amplitude: list[ParameterType]
    delta: list[ParameterType]

    def calculate_matrix(
        self,
        dataset_model,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        location = np.asarray(self.location)
        amp = np.asarray(self.amplitude)
        delta = np.asarray(self.delta)

        array = np.empty((location.size, model_axis.size), dtype=np.float64)

        for i in range(location.size):
            array[i, :] = amp[i] * np.exp(
                -np.log(2) * np.square(2 * (model_axis - location[i]) / delta[i])
            )
        compartments = [f"s{i+1}" for i in range(location.size)]
        return compartments, array.T

    def finalize_data(
        self,
        dataset_model,
        dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


DecayModel = Model.create_class_from_megacomplexes(
    [SimpleKineticMegacomplex, SimpleSpectralMegacomplex, ShapedSpectralMegacomplex]
)
