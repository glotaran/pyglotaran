from __future__ import annotations

import numpy as np
import pytest

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.builtin.megacomplexes.damped_oscillation import DampedOscillationMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.builtin.megacomplexes.spectral import SpectralMegacomplex
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


class DampedOscillationsModel(Model):
    @classmethod
    def from_dict(
        cls,
        model_dict,
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = None,
        default_megacomplex_type: str | None = None,
    ):
        defaults: dict[str, type[Megacomplex]] = {
            "damped_oscillation": DampedOscillationMegacomplex,
            "decay": DecayMegacomplex,
            "spectral": SpectralMegacomplex,
        }
        if megacomplex_types is not None:
            defaults.update(megacomplex_types)
        return super().from_dict(
            model_dict,
            megacomplex_types=defaults,
            default_megacomplex_type=default_megacomplex_type,
        )


class OneOscillation:
    sim_model = DampedOscillationsModel.from_dict(
        {
            "megacomplex": {
                "m1": {
                    "type": "damped_oscillation",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
                "m2": {
                    "type": "spectral",
                    "shape": {"osc1_cos": "sh1", "osc1_sin": "sh1"},
                },
            },
            "shape": {
                "sh1": {
                    "type": "gaussian",
                    "amplitude": "shapes.amps.1",
                    "location": "shapes.locs.1",
                    "width": "shapes.width.1",
                },
            },
            "dataset": {"dataset1": {"megacomplex": ["m1"], "global_megacomplex": ["m2"]}},
        }
    )

    model = DampedOscillationsModel.from_dict(
        {
            "megacomplex": {
                "m1": {
                    "type": "damped_oscillation",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
            },
            "dataset": {"dataset1": {"megacomplex": ["m1"]}},
        }
    )

    wanted_parameter = ParameterGroup.from_dict(
        {
            "osc": [
                ["freq", 25.5],
                ["rate", 0.1],
            ],
            "shapes": {"amps": [7], "locs": [5], "width": [4]},
        }
    )

    parameter = ParameterGroup.from_dict(
        {
            "osc": [
                ["freq", 20],
                ["rate", 0.3],
            ],
        }
    )

    time = np.arange(0, 3, 0.01)
    spectral = np.arange(0, 10)
    axis = {"time": time, "spectral": spectral}

    wanted_clp = ["osc1_cos", "osc1_sin"]
    wanted_shape = (300, 2)


class OneOscillationWithIrf:
    sim_model = DampedOscillationsModel.from_dict(
        {
            "megacomplex": {
                "m1": {
                    "type": "damped_oscillation",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
                "m2": {
                    "type": "spectral",
                    "shape": {"osc1_cos": "sh1", "osc1_sin": "sh1"},
                },
            },
            "shape": {
                "sh1": {
                    "type": "gaussian",
                    "amplitude": "shapes.amps.1",
                    "location": "shapes.locs.1",
                    "width": "shapes.width.1",
                },
            },
            "irf": {
                "irf1": {
                    "type": "gaussian",
                    "center": "irf.center",
                    "width": "irf.width",
                },
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "irf": "irf1",
                }
            },
        }
    )

    model = DampedOscillationsModel.from_dict(
        {
            "megacomplex": {
                "m1": {
                    "type": "damped_oscillation",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
            },
            "irf": {
                "irf1": {
                    "type": "gaussian",
                    "center": "irf.center",
                    "width": "irf.width",
                },
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "irf": "irf1",
                }
            },
        }
    )

    wanted_parameter = ParameterGroup.from_dict(
        {
            "osc": [
                ["freq", 25],
                ["rate", 0.1],
            ],
            "shapes": {"amps": [7], "locs": [5], "width": [4]},
            "irf": [["center", 0.3], ["width", 0.1]],
        }
    )

    parameter = ParameterGroup.from_dict(
        {
            "osc": [
                ["freq", 25],
                ["rate", 0.1],
            ],
            "irf": [["center", 0.3], ["width", 0.1]],
        }
    )

    time = np.arange(0, 3, 0.01)
    spectral = np.arange(0, 10)
    axis = {"time": time, "spectral": spectral}

    wanted_clp = ["osc1_cos", "osc1_sin"]
    wanted_shape = (300, 2)


class OneOscillationWithSequentialModel:
    sim_model = DampedOscillationsModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1", "s2"], "parameters": ["j.1", "j.0"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s2", "s1"): "kinetic.1",
                        ("s2", "s2"): "kinetic.2",
                    }
                }
            },
            "megacomplex": {
                "m1": {"type": "decay", "k_matrix": ["k1"]},
                "m2": {
                    "type": "damped_oscillation",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
                "m3": {
                    "type": "spectral",
                    "shape": {
                        "osc1_cos": "sh1",
                        "osc1_sin": "sh1",
                        "s1": "sh2",
                        "s2": "sh3",
                    },
                },
            },
            "shape": {
                "sh1": {
                    "type": "gaussian",
                    "amplitude": "shapes.amps.1",
                    "location": "shapes.locs.1",
                    "width": "shapes.width.1",
                },
                "sh2": {
                    "type": "gaussian",
                    "amplitude": "shapes.amps.2",
                    "location": "shapes.locs.2",
                    "width": "shapes.width.2",
                },
                "sh3": {
                    "type": "gaussian",
                    "amplitude": "shapes.amps.3",
                    "location": "shapes.locs.3",
                    "width": "shapes.width.3",
                },
            },
            "irf": {
                "irf1": {
                    "type": "gaussian",
                    "center": "irf.center",
                    "width": "irf.width",
                },
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "irf": "irf1",
                    "megacomplex": ["m1", "m2"],
                    "global_megacomplex": ["m3"],
                }
            },
        }
    )

    model = DampedOscillationsModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1", "s2"], "parameters": ["j.1", "j.0"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s2", "s1"): "kinetic.1",
                        ("s2", "s2"): "kinetic.2",
                    }
                }
            },
            "megacomplex": {
                "m1": {"type": "decay", "k_matrix": ["k1"]},
                "m2": {
                    "type": "damped_oscillation",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
            },
            "irf": {
                "irf1": {
                    "type": "gaussian",
                    "center": "irf.center",
                    "width": "irf.width",
                },
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "irf": "irf1",
                    "megacomplex": ["m1", "m2"],
                }
            },
        }
    )

    wanted_parameter = ParameterGroup.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
                ["0", 0, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [
                ["1", 0.2],
                ["2", 0.01],
            ],
            "osc": [
                ["freq", 25],
                ["rate", 0.1],
            ],
            "shapes": {"amps": [0.07, 2, 4], "locs": [5, 2, 8], "width": [4, 2, 3]},
            "irf": [["center", 0.3], ["width", 0.1]],
        }
    )

    parameter = ParameterGroup.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
                ["0", 0, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [
                ["1", 0.2],
                ["2", 0.01],
            ],
            "osc": [
                ["freq", 25],
                ["rate", 0.1],
            ],
            "irf": [["center", 0.3], ["width", 0.1]],
        }
    )

    time = np.arange(-1, 5, 0.01)
    spectral = np.arange(0, 10)
    axis = {"time": time, "spectral": spectral}

    wanted_clp = ["osc1_cos", "osc1_sin", "s1", "s2"]
    wanted_shape = (600, 4)


@pytest.mark.parametrize(
    "suite",
    [
        OneOscillation,
        OneOscillationWithIrf,
        OneOscillationWithSequentialModel,
    ],
)
def test_doas_model(suite):

    print(suite.sim_model.validate())
    assert suite.sim_model.valid()

    print(suite.model.validate())
    assert suite.model.valid()

    print(suite.sim_model.validate(suite.wanted_parameter))
    assert suite.sim_model.valid(suite.wanted_parameter)

    print(suite.model.validate(suite.parameter))
    assert suite.model.valid(suite.parameter)

    dataset = simulate(suite.sim_model, "dataset1", suite.wanted_parameter, suite.axis)
    print(dataset)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

    print(suite.parameter)
    print(suite.wanted_parameter)

    data = {"dataset1": dataset}
    scheme = Scheme(
        model=suite.model,
        parameters=suite.parameter,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme, raise_exception=True)
    print(result.optimized_parameters)

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, suite.wanted_parameter.get(label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data)

    assert "damped_oscillation_cos" in resultdata
    assert "damped_oscillation_sin" in resultdata
    assert "damped_oscillation_associated_spectra" in resultdata
    assert "damped_oscillation_phase" in resultdata
