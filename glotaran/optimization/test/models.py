from __future__ import annotations

from typing import List

import numpy as np

from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex
from glotaran.parameter import Parameter


@megacomplex(dimension="model", properties={"is_index_dependent": bool})
class SimpleTestMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_index: int | None,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):

        compartments = ["s1", "s2"]
        r_compartments = []
        array = np.zeros((model_axis.size, len(compartments)))

        for i in range(len(compartments)):
            r_compartments.append(compartments[i])
            for j in range(model_axis.size):
                array[j, i] = (i + j) * model_axis[j]
        return r_compartments, array

    def index_dependent(self, dataset_model):
        return self.is_index_dependent

    def finalize_data(
        self,
        dataset_model,
        dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


class SimpleTestModel(Model):
    @classmethod
    def from_dict(
        cls,
        model_dict,
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = None,
        default_megacomplex_type: str | None = None,
    ):
        defaults: dict[str, type[Megacomplex]] = {
            "model_complex": SimpleTestMegacomplex,
            #  "global_complex": SimpleTestMegacomplexGlobal,
        }
        if megacomplex_types is not None:
            defaults.update(megacomplex_types)
        return super().from_dict(
            model_dict,
            megacomplex_types=defaults,
            default_megacomplex_type=default_megacomplex_type,
        )


@megacomplex(
    dimension="model",
    properties={"is_index_dependent": bool},
    dataset_properties={
        "kinetic": List[Parameter],
    },
)
class SimpleKineticMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model,
        global_index: int | None,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):
        kinpar = -1 * np.asarray(dataset_model.kinetic)
        if dataset_model.label == "dataset3":
            # this case is for the ThreeDatasetDecay test
            compartments = [f"s{i+2}" for i in range(len(kinpar))]
        else:
            compartments = [f"s{i+1}" for i in range(len(kinpar))]
        array = np.exp(np.outer(model_axis, kinpar))
        return compartments, array

    def index_dependent(self, dataset_model):
        return self.is_index_dependent

    def finalize_data(
        self,
        dataset_model,
        dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


@megacomplex(dimension="global", properties={})
class SimpleSpectralMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model,
        global_index: int | None,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
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

    def index_dependent(self, dataset_model):
        return False


@megacomplex(
    dimension="global",
    properties={
        "location": {"type": List[Parameter], "allow_none": True},
        "amplitude": {"type": List[Parameter], "allow_none": True},
        "delta": {"type": List[Parameter], "allow_none": True},
    },
)
class ShapedSpectralMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        dataset_model,
        global_index: int | None,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
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

    def index_dependent(self, dataset_model):
        return False

    def finalize_data(
        self,
        dataset_model,
        dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


class DecayModel(Model):
    @classmethod
    def from_dict(
        cls,
        model_dict,
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = None,
        default_megacomplex_type: str | None = None,
    ):
        defaults: dict[str, type[Megacomplex]] = {
            "model_complex": SimpleKineticMegacomplex,
            "global_complex": SimpleSpectralMegacomplex,
            "global_complex_shaped": ShapedSpectralMegacomplex,
        }
        if megacomplex_types is not None:
            defaults.update(megacomplex_types)
        return super().from_dict(
            model_dict,
            megacomplex_types=defaults,
            default_megacomplex_type=default_megacomplex_type,
        )
