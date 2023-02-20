from __future__ import annotations

import numpy as np

from glotaran.optimization.test.models import DecayModel
from glotaran.parameter import Parameters


class OneCompartmentDecay:
    scale = 2
    wanted_parameters = Parameters.from_list([101e-4])
    initial_parameters = Parameters.from_list([100e-5, [scale, {"vary": False}]])

    global_axis = np.asarray([1.0])
    model_axis = np.arange(0, 150, 1.5)

    sim_model_dict = {
        "megacomplex": {
            "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
            "m2": {"type": "simple-spectral-test-mc"},
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1"],
            }
        },
    }
    sim_model = DecayModel(**sim_model_dict)
    model_dict = {
        "megacomplex": {"m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False}},
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
                "kinetic": ["1"],
                "scale": "2",
            }
        },
    }
    model_dict["dataset"]["dataset1"]["scale"] = "2"  # type:ignore[index]
    model = DecayModel(**model_dict)


class TwoCompartmentDecay:
    wanted_parameters = Parameters.from_list([11e-4, 22e-5])
    initial_parameters = Parameters.from_list([10e-4, 20e-5])

    global_axis = np.asarray([1.0, 2])
    model_axis = np.arange(0, 150, 1.5)

    sim_model = DecayModel(
        **{
            "megacomplex": {
                "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
                "m2": {"type": "simple-spectral-test-mc"},
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "kinetic": ["1", "2"],
                }
            },
        }
    )
    model = DecayModel(
        **{
            "megacomplex": {"m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "kinetic": ["1", "2"],
                }
            },
        }
    )


class ThreeDatasetDecay:
    wanted_parameters = Parameters.from_list([101e-4, 201e-3])
    initial_parameters = Parameters.from_list([100e-5, 200e-3])

    global_axis = np.asarray([1.0])
    model_axis = np.arange(0, 150, 1.5)

    global_axis2 = np.asarray([1.0, 2.01])
    model_axis2 = np.arange(0, 100, 1.5)

    global_axis3 = np.asarray([0.99, 3.0])
    model_axis3 = np.arange(0, 150, 1.5)

    sim_model_dict = {
        "megacomplex": {
            "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
            "m2": {"type": "simple-spectral-test-mc"},
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1"],
            },
            "dataset2": {
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1", "2"],
            },
            "dataset3": {
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["2"],
            },
        },
    }
    sim_model = DecayModel(**sim_model_dict)

    model_dict = {
        "megacomplex": {"m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False}},
        "dataset": {
            "dataset1": {"megacomplex": ["m1"], "kinetic": ["1"]},
            "dataset2": {
                "megacomplex": ["m1"],
                "kinetic": ["1", "2"],
            },
            "dataset3": {"megacomplex": ["m1"], "kinetic": ["2"]},
        },
    }
    model = DecayModel(**model_dict)


class MultichannelMulticomponentDecay:
    wanted_parameters = Parameters.from_dict(
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
    initial_parameters = Parameters.from_dict({"k": [0.006, 0.003, 0.0003, 0.03]})

    global_axis = np.arange(12820, 15120, 50)
    model_axis = np.arange(0, 150, 1.5)

    sim_model = DecayModel(
        **{
            "megacomplex": {
                "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
                "m2": {
                    "type": "shaped-spectral-test-mc",
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
    model = DecayModel(
        **{
            "megacomplex": {"m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "kinetic": ["k.1", "k.2", "k.3", "k.4"],
                }
            },
        }
    )


class FullModel:
    model = DecayModel(
        **{
            "megacomplex": {
                "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
                "m2": {
                    "type": "shaped-spectral-test-mc",
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
    parameters = Parameters.from_dict(
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
