from __future__ import annotations

import numpy as np
import pytest

from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.builtin.megacomplexes.pfid import PFIDMegacomplex
from glotaran.builtin.megacomplexes.spectral import SpectralMegacomplex
from glotaran.model import Model
from glotaran.optimization.optimize import optimize
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


class OneOscillationWithIrf:
    pure_pfid_model = Model.create_class_from_megacomplexes([PFIDMegacomplex, SpectralMegacomplex])
    sim_model = pure_pfid_model(
        **{
            "megacomplex": {
                "pfid": {
                    "type": "pfid",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
                "spectral": {
                    "type": "spectral",
                    "shape": {
                        "osc1_cos": "sh2",
                        "osc1_sin": "sh1",
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
                    "megacomplex": ["pfid"],
                    "global_megacomplex": ["spectral"],
                    "irf": "irf1",
                }
            },
        }
    )

    model = pure_pfid_model(
        **{
            "megacomplex": {
                "m1": {
                    "type": "pfid",
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

    wanted_parameter = Parameters.from_dict(
        {
            "osc": [
                ["freq", 1500],
                ["rate", -2],
            ],
            "shapes": {"amps": [2, -2], "locs": [1490, 1510], "width": [4, 4]},
            "irf": [["center", 0.01], ["width", 0.05]],
        }
    )

    parameter = Parameters.from_dict(
        {
            "osc": [
                ["freq", 1501],
                ["rate", -2.1],
            ],
            "irf": [["center", 0.01], ["width", 0.05]],
        }
    )

    time = np.arange(-4, 1, 0.01)
    spectral = np.arange(1480, 1520, 1)
    axis = {"time": time, "spectral": spectral}

    wanted_clp = ["pfid1"]
    wanted_shape = (40, 1)


class OneOscillationWithSequentialModel:
    decay_pfid_model = Model.create_class_from_megacomplexes(
        [PFIDMegacomplex, DecayMegacomplex, SpectralMegacomplex]
    )
    sim_model = decay_pfid_model(
        **{
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
                    "type": "pfid",
                    "labels": ["osc1"],
                    "frequencies": ["osc.freq"],
                    "rates": ["osc.rate"],
                },
                "m3": {
                    "type": "spectral",
                    "shape": {
                        "s1": "sh3",
                        "s2": "sh4",
                        "osc1_cos": "sh2",
                        "osc1_sin": "sh1",
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
                "sh4": {
                    "type": "gaussian",
                    "amplitude": "shapes.amps.4",
                    "location": "shapes.locs.4",
                    "width": "shapes.width.4",
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
                    "megacomplex": ["m2", "m1"],
                    "global_megacomplex": ["m3"],
                }
            },
        }
    )

    model = decay_pfid_model(
        **{
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
                    "type": "pfid",
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

    wanted_parameter = Parameters.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
                ["0", 0, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [
                ["1", 0.05],
                ["2", 0.001],
            ],
            "osc": [
                ["freq", 1500],
                ["rate", -2],
            ],
            "shapes": {
                "amps": [2, -2, 8, 9],
                "locs": [1490, 1510, 1495, 1505],
                "width": [4, 4, 3, 5],
            },
            "irf": [["center", 0.01], ["width", 0.05]],
        }
    )

    parameter = Parameters.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
                ["0", 0, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [
                ["1", 0.055],
                ["2", 0.0015],
            ],
            "osc": [
                ["freq", 1501],
                ["rate", -2.1],
            ],
            "irf": [["center", 0.01], ["width", 0.05]],
        }
    )

    time = np.arange(-5, 80, 0.01)
    spectral = np.arange(1480, 1520, 1)
    axis = {"time": time, "spectral": spectral}

    wanted_clp = ["osc1_cos", "osc1_sin", "s1", "s2"]
    wanted_shape = (600, 4)


@pytest.mark.parametrize(
    "suite",
    [
        OneOscillationWithIrf,
        OneOscillationWithSequentialModel,
    ],
)
def test_pfid_model(suite):
    class_name = suite.__name__
    print(suite.sim_model.validate())
    assert suite.sim_model.valid()

    print(suite.model.validate())
    assert suite.model.valid()

    print(suite.sim_model.validate(suite.wanted_parameter))
    assert suite.sim_model.valid(suite.wanted_parameter)

    print(suite.model.validate(suite.parameter))
    assert suite.model.valid(suite.parameter)

    dataset = simulate(
        suite.sim_model,
        "dataset1",
        suite.wanted_parameter,
        suite.axis,
        noise=True,
        noise_std_dev=1e-8,
        noise_seed=123,
    )
    print(dataset)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

    print(suite.parameter)
    print(suite.wanted_parameter)

    data = {"dataset1": dataset}
    scheme = Scheme(
        model=suite.model,
        parameters=suite.parameter,
        data=data,
        maximum_number_function_evaluations=5,
    )
    result = optimize(scheme, raise_exception=True)
    print(result.optimized_parameters)

    for param in result.optimized_parameters.all():
        assert np.allclose(param.value, suite.wanted_parameter.get(param.label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, atol=1e-5)

    assert "pfid_associated_spectra" in resultdata
    assert "pfid_phase" in resultdata

    # Ensure that s1, s2 are not mixed up with osc1_cos and osc1_sin by checking amplitudes
    if "OneOscillationWithSequentialModel" in class_name:
        assert resultdata.species_associated_spectra.sel(species="s1").max() > 7
        assert resultdata.species_associated_spectra.sel(species="s2").max() > 8


def test_pfid_model_validate():
    """An ``OscillationParameterIssue`` should be raised if there is a list length mismatch.

    List values are: ``labels``, ``frequencies``, ``rates``.
    """
    pure_pfid_model = Model.create_class_from_megacomplexes([PFIDMegacomplex, SpectralMegacomplex])
    model_data = OneOscillationWithIrf.sim_model.as_dict()
    model_data["megacomplex"]["pfid"]["labels"].append("extra-label")
    model = pure_pfid_model(**model_data)
    validation_msg = model.validate()
    assert (
        validation_msg == "Your model has 1 problem:\n\n"
        " * The size of labels (2), frequencies (1), and rates (1) does not match for pfid "
        "megacomplex 'pfid'."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
