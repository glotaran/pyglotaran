import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.util import calculate_matrix
from glotaran.builtin.models.kinetic_image import KineticImageModel
from glotaran.builtin.models.spectral.spectral_model import SpectralModel
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


class OneCompartmentModel:
    kinetic_model = KineticImageModel.from_dict(
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

    kinetic_parameters = ParameterGroup.from_list(
        [101e-4, [1, {"vary": False, "non-negative": False}]]
    )

    spectral_model = SpectralModel.from_dict(
        {
            "megacomplex": {
                "mc1": {"shape": {"s1": "sh1"}},
            },
            "shape": {
                "sh1": {
                    "type": "skewed-gaussian",
                    "amplitude": "1",
                    "location": "2",
                    "width": "3",
                }
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    spectral_parameters = ParameterGroup.from_list([7, 20000, 800])

    time = np.arange(-10, 50, 1.5)
    spectral = np.arange(400, 600, 5)
    axis = {"time": time, "spectral": spectral}

    dataset = kinetic_model.dataset["dataset1"].fill(kinetic_model, kinetic_parameters)
    kinetic_compartments, kinetic_matrix = calculate_matrix(kinetic_model, dataset, {}, axis)
    clp = xr.DataArray(
        kinetic_matrix, coords=[("time", time), ("clp_label", kinetic_compartments)]
    )


class ThreeCompartmentModel:
    kinetic_model = KineticImageModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1", "s2", "s3"], "parameters": ["4", "4", "4"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "1",
                        ("s2", "s2"): "2",
                        ("s3", "s3"): "3",
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

    kinetic_parameters = ParameterGroup.from_list(
        [101e-4, 101e-5, 101e-6, [1, {"vary": False, "non-negative": False}]]
    )

    spectral_model = SpectralModel.from_dict(
        {
            "megacomplex": {
                "mc1": {
                    "shape": {
                        "s1": "sh1",
                        "s2": "sh2",
                        "s3": "sh3",
                    }
                },
            },
            "shape": {
                "sh1": {
                    "type": "skewed-gaussian",
                    "amplitude": "1",
                    "location": "2",
                    "width": "3",
                },
                "sh2": {
                    "type": "skewed-gaussian",
                    "amplitude": "4",
                    "location": "5",
                    "width": "6",
                },
                "sh3": {
                    "type": "skewed-gaussian",
                    "amplitude": "7",
                    "location": "8",
                    "width": "9",
                    "skewness": "10",
                },
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["mc1"],
                },
            },
        }
    )

    spectral_parameters = ParameterGroup.from_list(
        [
            7,
            20000,
            800,
            20,
            22000,
            500,
            10,
            18000,
            650,
            0.1,
        ]
    )

    time = np.arange(-10, 50, 1.5)
    spectral = np.arange(400, 600, 5)
    axis = {"time": time, "spectral": spectral}

    dataset = kinetic_model.dataset["dataset1"].fill(kinetic_model, kinetic_parameters)
    kinetic_compartments, kinetic_matrix = calculate_matrix(kinetic_model, dataset, {}, axis)
    clp = xr.DataArray(
        kinetic_matrix, coords=[("time", time), ("clp_label", kinetic_compartments)]
    )


@pytest.mark.parametrize(
    "suite",
    [
        OneCompartmentModel,
        ThreeCompartmentModel,
    ],
)
def test_spectral_model(suite):

    model = suite.spectral_model
    print(model.validate())
    assert model.valid()

    wanted_parameters = suite.spectral_parameters
    print(model.validate(wanted_parameters))
    print(wanted_parameters)
    assert model.valid(wanted_parameters)

    initial_parameters = suite.spectral_parameters
    print(model.validate(initial_parameters))
    assert model.valid(initial_parameters)

    print(model.markdown(initial_parameters))

    dataset = simulate(model, "dataset1", wanted_parameters, suite.axis, suite.clp)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

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
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)
    assert "species_associated_concentrations" in resultdata
    assert resultdata.species_associated_concentrations.shape == (
        suite.axis["time"].size,
        len(suite.kinetic_compartments),
    )
    assert "species_spectra" in resultdata
    assert resultdata.species_spectra.shape == (
        suite.axis["spectral"].size,
        len(suite.kinetic_compartments),
    )
