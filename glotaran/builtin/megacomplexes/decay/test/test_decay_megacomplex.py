from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayParallelMegacomplex
from glotaran.builtin.megacomplexes.decay import DecaySequentialMegacomplex
from glotaran.model import Model
from glotaran.optimization.optimize import optimize
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


def create_gaussian_clp(labels, amplitudes, centers, widths, axis):
    return xr.DataArray(
        [
            amplitudes[i] * np.exp(-np.log(2) * np.square(2 * (axis - centers[i]) / widths[i]))
            for i, _ in enumerate(labels)
        ],
        coords=[("clp_label", labels), ("pixel", axis.data)],
    ).T


DecaySimpleModel = Model.create_class_from_megacomplexes(
    [DecayParallelMegacomplex, DecaySequentialMegacomplex]
)
DecayModel = Model.create_class_from_megacomplexes([DecayMegacomplex])


class OneComponentOneChannel:
    model = DecayModel(
        **{
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["2"]},
            },
            "megacomplex": {
                "mc1": {"type": "decay", "k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "1",
                    }
                }
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    initial_parameters = Parameters.from_list(
        [101e-4, [1, {"vary": False, "non-negative": False}]]
    )
    wanted_parameters = Parameters.from_list([101e-3, [1, {"vary": False, "non-negative": False}]])

    time = np.arange(0, 50, 1.5)
    pixel = np.asarray([0])
    axis = {"time": time, "pixel": pixel}

    clp = xr.DataArray([[1]], coords=[("pixel", pixel.data), ("clp_label", ["s1"])])


class OneComponentOneChannelGaussianIrf:
    model = DecayModel(
        **{
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["5"]},
            },
            "megacomplex": {
                "mc1": {"type": "decay", "k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "1",
                    }
                }
            },
            "irf": {
                "irf1": {"type": "gaussian", "center": "2", "width": "3", "shift": ["4"]},
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "irf": "irf1",
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    initial_parameters = Parameters.from_list(
        [
            101e-4,
            0.1,
            1,
            [0.1, {"vary": False}],
            [1, {"vary": False, "non-negative": False}],
        ]
    )
    print(initial_parameters)
    wanted_parameters = Parameters.from_list(
        [
            [101e-3, {"non-negative": True}],
            [0.2, {"non-negative": True}],
            [2, {"non-negative": True}],
            [0.1, {"vary": False}],
            [1, {"vary": False, "non-negative": False}],
        ]
    )

    time = np.arange(0, 50, 1.5)
    pixel = np.asarray([0])
    axis = {"time": time, "pixel": pixel}

    clp = xr.DataArray([[1]], coords=[("pixel", pixel.data), ("clp_label", ["s1"])])


class ThreeComponentParallel:
    model = DecaySimpleModel(
        **{
            "megacomplex": {
                "mc1": {
                    "type": "decay-parallel",
                    "compartments": ["s1", "s2", "s3"],
                    "rates": [
                        "kinetic.1",
                        "kinetic.2",
                        "kinetic.3",
                    ],
                },
            },
            "irf": {
                "irf1": {
                    "type": "multi-gaussian",
                    "center": ["irf.center"],
                    "width": ["irf.width"],
                },
            },
            "dataset": {
                "dataset1": {
                    "irf": "irf1",
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    initial_parameters = Parameters.from_dict(
        {
            "kinetic": [
                ["1", 501e-3],
                ["2", 202e-4],
                ["3", 105e-5],
                {"non-negative": True},  # type: ignore[list-item]
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
        }
    )
    wanted_parameters = Parameters.from_dict(
        {
            "kinetic": [
                ["1", 501e-3],
                ["2", 202e-4],
                ["3", 105e-5],
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
        }
    )

    time = np.arange(-10, 100, 1.5)
    pixel = np.arange(600, 750, 10)

    axis = {"time": time, "pixel": pixel}

    clp = create_gaussian_clp(["s1", "s2", "s3"], [7, 3, 30], [620, 670, 720], [10, 30, 50], pixel)


class ThreeComponentSequential:
    model = DecaySimpleModel(
        **{
            "megacomplex": {
                "mc1": {
                    "type": "decay-sequential",
                    "compartments": ["s1", "s2", "s3"],
                    "rates": [
                        "kinetic.1",
                        "kinetic.2",
                        "kinetic.3",
                    ],
                },
            },
            "irf": {
                "irf1": {
                    "type": "multi-gaussian",
                    "center": ["irf.center"],
                    "width": ["irf.width"],
                },
            },
            "dataset": {
                "dataset1": {
                    "irf": "irf1",
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    initial_parameters = Parameters.from_dict(
        {
            "kinetic": [
                ["1", 501e-3],
                ["2", 202e-4],
                ["3", 105e-5],
                {"non-negative": True},  # type: ignore[list-item]
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
        }
    )
    wanted_parameters = Parameters.from_dict(
        {
            "kinetic": [
                ["1", 501e-3],
                ["2", 202e-4],
                ["3", 105e-5],
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
        }
    )

    time = np.arange(-10, 50, 1.0)
    pixel = np.arange(600, 750, 10)
    axis = {"time": time, "pixel": pixel}

    clp = create_gaussian_clp(["s1", "s2", "s3"], [7, 3, 30], [620, 670, 720], [10, 30, 50], pixel)


@pytest.mark.parametrize(
    "suite",
    [
        OneComponentOneChannel,
        OneComponentOneChannelGaussianIrf,
        ThreeComponentParallel,
        ThreeComponentSequential,
    ],
)
@pytest.mark.parametrize("nnls", [True, False])
def test_kinetic_model(suite, nnls):
    model = suite.model
    print(model.validate())
    assert model.valid()
    model.dataset_groups["default"].method = (
        "non_negative_least_squares" if nnls else "variable_projection"
    )

    wanted_parameters = suite.wanted_parameters
    print(model.validate(wanted_parameters))
    print(wanted_parameters)
    assert model.valid(wanted_parameters)

    initial_parameters = suite.initial_parameters
    print(model.validate(initial_parameters))
    assert model.valid(initial_parameters)

    print(model.markdown(initial_parameters))

    dataset = simulate(model, "dataset1", wanted_parameters, suite.axis, suite.clp)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["pixel"].size)

    data = {"dataset1": dataset}

    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for param in result.optimized_parameters.all():
        assert np.allclose(param.value, wanted_parameters.get(param.label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["pixel"], resultdata["pixel"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)

    if suite is OneComponentOneChannelGaussianIrf:
        assert "irf_shift" in resultdata
        expected_center = wanted_parameters.get("2").value
        expected_shift = wanted_parameters.get("4").value
        assert np.allclose(resultdata["irf_shift"].values, expected_center - expected_shift)

    if len(model.irf) != 0:
        assert "irf" in resultdata


def test_finalize_data():
    model = DecayModel(
        **{
            "initial_concentration": {
                "j1": {"compartments": ["s1", "s2"], "parameters": ["3", "3"]},
            },
            "megacomplex": {
                "mc1": {"type": "decay", "k_matrix": ["k1"]},
                "mc2": {"type": "decay", "k_matrix": ["k2"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "1",
                    }
                },
                "k2": {
                    "matrix": {
                        ("s2", "s2"): "2",
                    }
                },
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "megacomplex": ["mc1", "mc2"],
                },
            },
        }
    )

    parameters = Parameters.from_list(
        [101e-4, 101e-3, [1, {"vary": False, "non-negative": False}]]
    )

    time = np.arange(0, 50, 1.5)
    pixel = np.asarray([0, 2])
    axis = {"time": time, "pixel": pixel}

    clp = xr.DataArray(
        [[1, 0], [0, 1]], coords=[("pixel", pixel.data), ("clp_label", ["s1", "s2"])]
    )
    dataset = simulate(model, "dataset1", parameters, axis, clp)

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data={"dataset1": dataset},
        maximum_number_function_evaluations=1,
    )
    result = optimize(scheme)

    result_dataset = result.data["dataset1"]

    assert "initial_concentration" in result_dataset
    assert "species_concentration" in result_dataset
    assert "species_associated_images" in result_dataset

    assert "decay_associated_images_mc1" in result_dataset
    assert "decay_associated_images_mc2" in result_dataset

    assert "k_matrix_mc1" in result_dataset
    assert "k_matrix_mc2" in result_dataset
    assert "k_matrix_reduced_mc1" in result_dataset
    assert "k_matrix_reduced_mc2" in result_dataset
    assert "a_matrix_mc1" in result_dataset
    assert "a_matrix_mc2" in result_dataset

    assert "species_mc1" in result_dataset.coords
    assert "species_mc2" in result_dataset.coords
    assert "initial_concentration_mc1" in result_dataset.coords
    assert "initial_concentration_mc2" in result_dataset.coords
    assert "component_mc1" in result_dataset.coords
    assert "component_mc2" in result_dataset.coords
    assert "rate_mc1" in result_dataset.coords
    assert "rate_mc2" in result_dataset.coords
    assert "lifetime_mc1" in result_dataset.coords
    assert "lifetime_mc2" in result_dataset.coords
