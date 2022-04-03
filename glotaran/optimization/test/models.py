from __future__ import annotations

from typing import List

import numpy as np

from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup


@megacomplex(dimension="global", properties={})
class SimpleTestMegacomplexGlobal(Megacomplex):
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        axis = dataset_model.get_coordinates()
        assert "model" in axis
        assert "global" in axis
        axis = axis["global"]
        compartments = ["s1", "s2"]
        r_compartments = []
        array = np.zeros((axis.shape[0], len(compartments)))

        for i in range(len(compartments)):
            r_compartments.append(compartments[i])
            for j in range(axis.shape[0]):
                array[j, i] = (i + j) * axis[j]
        return r_compartments, array

    def index_dependent(self, dataset_model):
        return False


@megacomplex(dimension="model", properties={"is_index_dependent": bool})
class SimpleTestMegacomplex(Megacomplex):
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        axis = dataset_model.get_coordinates()
        assert "model" in axis
        assert "global" in axis

        axis = axis["model"]
        compartments = ["s1", "s2"]
        r_compartments = []
        array = np.zeros((axis.shape[0], len(compartments)))

        for i in range(len(compartments)):
            r_compartments.append(compartments[i])
            for j in range(axis.shape[0]):
                array[j, i] = (i + j) * axis[j]
        return r_compartments, array

    def index_dependent(self, dataset_model):
        return self.is_index_dependent


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
            "global_complex": SimpleTestMegacomplexGlobal,
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
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        axis = dataset_model.get_coordinates()
        axis = axis["model"]
        kinpar = -1 * np.asarray(dataset_model.kinetic)
        if dataset_model.label == "dataset3":
            # this case is for the ThreeDatasetDecay test
            compartments = [f"s{i+2}" for i in range(len(kinpar))]
        else:
            compartments = [f"s{i+1}" for i in range(len(kinpar))]
        array = np.exp(np.outer(axis, kinpar))
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
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        axis = dataset_model.get_coordinates()
        axis = axis["global"]
        kinpar = dataset_model.kinetic
        if dataset_model.label == "dataset3":
            # this case is for the ThreeDatasetDecay test
            compartments = [f"s{i+2}" for i in range(len(kinpar))]
        else:
            compartments = [f"s{i+1}" for i in range(len(kinpar))]
        array = np.asarray([[1 for _ in range(axis.size)] for _ in compartments]).T
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
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        location = np.asarray(self.location)
        amp = np.asarray(self.amplitude)
        delta = np.asarray(self.delta)

        axis = dataset_model.get_coordinates()
        axis = axis["global"]
        array = np.empty((location.size, axis.size), dtype=np.float64)

        for i in range(location.size):
            array[i, :] = amp[i] * np.exp(
                -np.log(2) * np.square(2 * (axis - location[i]) / delta[i])
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


class OneCompartmentDecay:
    scale = 2
    wanted_parameters = ParameterGroup.from_list([101e-4])
    initial_parameters = ParameterGroup.from_list([100e-5, [scale, {"vary": False}]])

    global_axis = np.asarray([1.0])
    model_axis = np.arange(0, 150, 1.5)

    sim_model_dict = {
        "megacomplex": {"m1": {"is_index_dependent": False}, "m2": {"type": "global_complex"}},
        "dataset": {
            "dataset1": {
                "initial_concentration": [],
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1"],
            }
        },
    }
    sim_model = DecayModel.from_dict(sim_model_dict)
    model_dict = {
        "megacomplex": {"m1": {"is_index_dependent": False}},
        "dataset": {
            "dataset1": {
                "initial_concentration": [],
                "megacomplex": ["m1"],
                "kinetic": ["1"],
                "scale": "2",
            }
        },
    }
    model_dict["dataset"]["dataset1"]["scale"] = "2"
    model = DecayModel.from_dict(model_dict)


class TwoCompartmentDecay:
    wanted_parameters = ParameterGroup.from_list([11e-4, 22e-5])
    initial_parameters = ParameterGroup.from_list([10e-4, 20e-5])

    global_axis = np.asarray([1.0])
    model_axis = np.arange(0, 150, 1.5)

    sim_model = DecayModel.from_dict(
        {
            "megacomplex": {"m1": {"is_index_dependent": False}, "m2": {"type": "global_complex"}},
            "dataset": {
                "dataset1": {
                    "initial_concentration": [],
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "kinetic": ["1", "2"],
                }
            },
        }
    )
    model = DecayModel.from_dict(
        {
            "megacomplex": {"m1": {"is_index_dependent": False}},
            "dataset": {
                "dataset1": {
                    "initial_concentration": [],
                    "megacomplex": ["m1"],
                    "kinetic": ["1", "2"],
                }
            },
        }
    )


class ThreeDatasetDecay:
    wanted_parameters = ParameterGroup.from_list([101e-4, 201e-3])
    initial_parameters = ParameterGroup.from_list([100e-5, 200e-3])

    global_axis = np.asarray([1.0])
    model_axis = np.arange(0, 150, 1.5)

    global_axis2 = np.asarray([1.0, 2.01])
    model_axis2 = np.arange(0, 100, 1.5)

    global_axis3 = np.asarray([0.99, 3.0])
    model_axis3 = np.arange(0, 150, 1.5)

    sim_model_dict = {
        "megacomplex": {"m1": {"is_index_dependent": False}, "m2": {"type": "global_complex"}},
        "dataset": {
            "dataset1": {
                "initial_concentration": [],
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1"],
            },
            "dataset2": {
                "initial_concentration": [],
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1", "2"],
            },
            "dataset3": {
                "initial_concentration": [],
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["2"],
            },
        },
    }
    sim_model = DecayModel.from_dict(sim_model_dict)

    model_dict = {
        "megacomplex": {"m1": {"is_index_dependent": False}},
        "dataset": {
            "dataset1": {"initial_concentration": [], "megacomplex": ["m1"], "kinetic": ["1"]},
            "dataset2": {
                "initial_concentration": [],
                "megacomplex": ["m1"],
                "kinetic": ["1", "2"],
            },
            "dataset3": {"initial_concentration": [], "megacomplex": ["m1"], "kinetic": ["2"]},
        },
    }
    model = DecayModel.from_dict(model_dict)


class MultichannelMulticomponentDecay:
    wanted_parameters = ParameterGroup.from_dict(
        {
            "k": [0.006, 0.003, 0.0003, 0.03],
            "loc": [
                ["1", 14705],
                ["2", 13513],
                ["3", 14492],
                ["4", 14388],
            ],
            "amp": [
                ["1", 1],
                ["2", 2],
                ["3", 5],
                ["4", 20],
            ],
            "del": [
                ["1", 400],
                ["2", 100],
                ["3", 300],
                ["4", 200],
            ],
        }
    )
    initial_parameters = ParameterGroup.from_dict({"k": [0.006, 0.003, 0.0003, 0.03]})

    global_axis = np.arange(12820, 15120, 50)
    model_axis = np.arange(0, 150, 1.5)

    sim_model = DecayModel.from_dict(
        {
            "megacomplex": {
                "m1": {"is_index_dependent": False},
                "m2": {
                    "type": "global_complex_shaped",
                    "location": ["loc.1", "loc.2", "loc.3", "loc.4"],
                    "delta": ["del.1", "del.2", "del.3", "del.4"],
                    "amplitude": ["amp.1", "amp.2", "amp.3", "amp.4"],
                },
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "kinetic": ["k.1", "k.2", "k.3", "k.4"],
                }
            },
        }
    )
    model = DecayModel.from_dict(
        {
            "megacomplex": {"m1": {"is_index_dependent": False}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "kinetic": ["k.1", "k.2", "k.3", "k.4"],
                }
            },
        }
    )


class FullModel:
    model = DecayModel.from_dict(
        {
            "megacomplex": {
                "m1": {"is_index_dependent": False},
                "m2": {
                    "type": "global_complex_shaped",
                    "location": ["loc.1", "loc.2", "loc.3", "loc.4"],
                    "delta": ["del.1", "del.2", "del.3", "del.4"],
                    "amplitude": ["amp.1", "amp.2", "amp.3", "amp.4"],
                },
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "kinetic": ["k.1", "k.2", "k.3", "k.4"],
                }
            },
        }
    )
    parameters = ParameterGroup.from_dict(
        {
            "k": [0.006, 0.003, 0.0003, 0.03],
            "loc": [
                ["1", 14705],
                ["2", 13513],
                ["3", 14492],
                ["4", 14388],
            ],
            "amp": [
                ["1", 1],
                ["2", 2],
                ["3", 5],
                ["4", 20],
            ],
            "del": [
                ["1", 400],
                ["2", 100],
                ["3", 300],
                ["4", 200],
            ],
        }
    )
    global_axis = np.arange(12820, 15120, 50)
    model_axis = np.arange(0, 150, 1.5)
    coordinates = {"global": global_axis, "model": model_axis}
