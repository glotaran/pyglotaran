from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.util import calculate_matrix
from glotaran.builtin.megacomplexes.decay.test.test_decay_megacomplex import DecayModel
from glotaran.builtin.megacomplexes.spectral import SpectralMegacomplex
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


class SpectralModel(Model):
    @classmethod
    def from_dict(
        cls,
        model_dict,
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = None,
        default_megacomplex_type: str | None = None,
    ):
        defaults: dict[str, type[Megacomplex]] = {
            "spectral": SpectralMegacomplex,
        }
        if megacomplex_types is not None:
            defaults.update(megacomplex_types)
        return super().from_dict(
            model_dict,
            megacomplex_types=defaults,
            default_megacomplex_type=default_megacomplex_type,
        )


class OneCompartmentModelInvertedAxis:
    decay_model = DecayModel.from_dict(
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

    decay_parameters = ParameterGroup.from_list(
        [101e-4, [1, {"vary": False, "non-negative": False}]]
    )

    spectral_model = SpectralModel.from_dict(
        {
            "megacomplex": {
                "mc1": {"shape": {"s1": "sh1"}},
            },
            "shape": {
                "sh1": {
                    "type": "gaussian",
                    "amplitude": "1",
                    "location": "2",
                    "width": "3",
                }
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["mc1"],
                    "spectral_axis_scale": 1e7,
                    "spectral_axis_inverted": True,
                },
            },
        }
    )

    spectral_parameters = ParameterGroup.from_list([7, 1e7 / 10000, 800, -1])

    time = np.arange(-10, 50, 1.5)
    spectral = np.arange(5000, 15000, 20)
    axis = {"time": time, "spectral": spectral}

    decay_dataset_model = decay_model.dataset["dataset1"].fill(decay_model, decay_parameters)
    decay_dataset_model.overwrite_global_dimension("spectral")
    decay_dataset_model.set_coordinates(axis)
    matrix = calculate_matrix(decay_dataset_model, {})
    decay_compartments = matrix.clp_labels
    clp = xr.DataArray(matrix.matrix, coords=[("time", time), ("clp_label", decay_compartments)])


class OneCompartmentModelNegativeSkew:
    decay_model = DecayModel.from_dict(
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

    decay_parameters = ParameterGroup.from_list(
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
                    "location": "1",
                    "width": "2",
                    "skewness": "3",
                }
            },
            "dataset": {
                "dataset1": {"megacomplex": ["mc1"], "spectral_axis_scale": 2},
            },
        }
    )

    spectral_parameters = ParameterGroup.from_list([1000, 80, -1])

    time = np.arange(-10, 50, 1.5)
    spectral = np.arange(400, 600, 5)
    axis = {"time": time, "spectral": spectral}

    decay_dataset_model = decay_model.dataset["dataset1"].fill(decay_model, decay_parameters)
    decay_dataset_model.overwrite_global_dimension("spectral")
    decay_dataset_model.set_coordinates(axis)
    matrix = calculate_matrix(decay_dataset_model, {})
    decay_compartments = matrix.clp_labels
    clp = xr.DataArray(matrix.matrix, coords=[("time", time), ("clp_label", decay_compartments)])


class OneCompartmentModelPositivSkew(OneCompartmentModelNegativeSkew):
    spectral_parameters = ParameterGroup.from_list([7, 20000, 800, 1])


class OneCompartmentModelZeroSkew(OneCompartmentModelNegativeSkew):
    spectral_parameters = ParameterGroup.from_list([7, 20000, 800, 0])


class ThreeCompartmentModel:
    decay_model = DecayModel.from_dict(
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

    decay_parameters = ParameterGroup.from_list(
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
                    "type": "gaussian",
                    "amplitude": "1",
                    "location": "2",
                    "width": "3",
                },
                "sh2": {
                    "type": "gaussian",
                    "amplitude": "4",
                    "location": "5",
                    "width": "6",
                },
                "sh3": {
                    "type": "gaussian",
                    "amplitude": "7",
                    "location": "8",
                    "width": "9",
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
            450,
            80,
            20,
            550,
            50,
            10,
            580,
            10,
        ]
    )

    time = np.arange(-10, 50, 1.5)
    spectral = np.arange(400, 600, 5)
    axis = {"time": time, "spectral": spectral}

    decay_dataset_model = decay_model.dataset["dataset1"].fill(decay_model, decay_parameters)
    decay_dataset_model.overwrite_global_dimension("spectral")
    decay_dataset_model.set_coordinates(axis)
    matrix = calculate_matrix(decay_dataset_model, {})
    decay_compartments = matrix.clp_labels
    clp = xr.DataArray(matrix.matrix, coords=[("time", time), ("clp_label", decay_compartments)])


@pytest.mark.parametrize(
    "suite",
    [
        OneCompartmentModelNegativeSkew,
        OneCompartmentModelPositivSkew,
        OneCompartmentModelZeroSkew,
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
        len(suite.decay_compartments),
    )
    assert "species_spectra" in resultdata
    assert resultdata.species_spectra.shape == (
        suite.axis["spectral"].size,
        len(suite.decay_compartments),
    )
