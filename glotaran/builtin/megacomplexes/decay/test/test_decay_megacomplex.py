from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme
from glotaran.testing.model_generators import SimpleModelGenerator


def _create_gaussian_clp(labels, amplitudes, centers, widths, axis):
    return xr.DataArray(
        [
            amplitudes[i] * np.exp(-np.log(2) * np.square(2 * (axis - centers[i]) / widths[i]))
            for i, _ in enumerate(labels)
        ],
        coords=[("clp_label", labels), ("pixel", axis.data)],
    ).T


class DecayModel(Model):
    @classmethod
    def from_dict(
        cls,
        model_dict,
    ):
        model_dict = {**model_dict, "default_megacomplex": "decay"}
        return super().from_dict(model_dict)


class OneComponentOneChannel:
    model = DecayModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["2"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
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

    initial_parameters = ParameterGroup.from_list(
        [101e-4, [1, {"vary": False, "non-negative": False}]]
    )
    wanted_parameters = ParameterGroup.from_list(
        [101e-3, [1, {"vary": False, "non-negative": False}]]
    )

    time = np.arange(0, 50, 1.5)
    pixel = np.asarray([0])
    axis = {"time": time, "pixel": pixel}

    clp = xr.DataArray([[1]], coords=[("pixel", pixel.data), ("clp_label", ["s1"])])


class OneComponentOneChannelGaussianIrf:
    model = DecayModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["5"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
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

    initial_parameters = ParameterGroup.from_list(
        [101e-4, 0.1, 1, [0.1, {"vary": False}], [1, {"vary": False, "non-negative": False}]]
    )
    assert model.megacomplex["mc1"].index_dependent(
        model.dataset["dataset1"].fill(model, initial_parameters)
    )
    wanted_parameters = ParameterGroup.from_list(
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
    generator = SimpleModelGenerator(
        rates=[300e-3, 500e-4, 700e-5],
        irf={"center": 1.3, "width": 7.8},
        k_matrix="parallel",
    )
    model, initial_parameters = generator.model_and_parameters

    generator.rates = [301e-3, 502e-4, 705e-5]
    wanted_parameters = generator.parameters

    time = np.arange(-10, 100, 1.5)
    pixel = np.arange(600, 750, 10)

    axis = {"time": time, "pixel": pixel}

    clp = _create_gaussian_clp(
        ["s1", "s2", "s3"], [7, 3, 30], [620, 670, 720], [10, 30, 50], pixel
    )


class ThreeComponentSequential:
    model = DecayModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1", "s2", "s3"], "parameters": ["j.1", "j.0", "j.0"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s2", "s1"): "kinetic.1",
                        ("s3", "s2"): "kinetic.2",
                        ("s3", "s3"): "kinetic.3",
                    }
                }
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
                    "initial_concentration": "j1",
                    "irf": "irf1",
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    initial_parameters = ParameterGroup.from_dict(
        {
            "kinetic": [
                ["1", 501e-3],
                ["2", 202e-4],
                ["3", 105e-5],
                {"non-negative": True},
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
                ["0", 0, {"vary": False, "non-negative": False}],
            ],
        }
    )
    wanted_parameters = ParameterGroup.from_dict(
        {
            "kinetic": [
                ["1", 501e-3],
                ["2", 202e-4],
                ["3", 105e-5],
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
                ["0", 0, {"vary": False, "non-negative": False}],
            ],
        }
    )

    time = np.arange(-10, 50, 1.0)
    pixel = np.arange(600, 750, 10)
    axis = {"time": time, "pixel": pixel}

    clp = _create_gaussian_clp(
        ["s1", "s2", "s3"], [7, 3, 30], [620, 670, 720], [10, 30, 50], pixel
    )


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
    model.dataset_group_models["default"].method = (
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

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, wanted_parameters.get(label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["pixel"], resultdata["pixel"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)
    assert "species_associated_images" in resultdata
    assert "decay_associated_images" in resultdata

    if len(model.irf) != 0:
        assert "irf" in resultdata
