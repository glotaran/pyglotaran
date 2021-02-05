from textwrap import dedent

import numpy as np
import pytest

from glotaran import read_model_from_yaml
from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.parameter import ParameterGroup


def get_dataset_block(megacomplex, input, irf, shape_tuples=None):
    line1 = "" if not shape_tuples else "shape:"
    shape_block = (
        "" if not shape_tuples else "".join(f"{key}: {value}\n" for key, value in shape_tuples)
    )
    return dedent(
        f"""\
    dataset:
        dataset1:
            megacomplex: {megacomplex}
            initial_concentration: {input}
            irf: {irf}
            {line1}
                {shape_block}
    """
    )


def get_shape_block(shape_key, type, amplitude=None, location=None, width=None):
    line1 = "" if not amplitude else f"amplitude: {amplitude}"
    line2 = "" if not location else f"location: {location}"
    line3 = "" if not width else f"width: {width}"
    return dedent(
        f"""\
        shape:
            {shape_key}:
                type: {type}
                {line1}
                {line2}
                {line3}
                """
    )


def get_initial_concentration_block(input_label, compartments, parameters):
    return dedent(
        f"""\
        initial_concentration:
            {input_label}:
                compartments: {compartments}
                parameters: {parameters}
    """
    )


def get_megacomplex_block(megacomplex_key, k_matrix):
    return dedent(
        f"""\
        megacomplex:
            {megacomplex_key}:
                k_matrix: {k_matrix}
    """
    )


def get_k_matrix_block(k_matrix_key, matrix_tuples):
    matrix_block = "".join(f"{key}: {value}\n" for key, value in matrix_tuples)
    return dedent(
        f"""\
        k_matrix:
            {k_matrix_key}:
                matrix:
                    {matrix_block}
    """
    )


def get_k_irf_block(
    irf_key,
    type,
    center,
    width,
    dispersion_center=None,
    center_dispersion=None,
    width_dispersion=None,
):
    line1 = "" if not dispersion_center else f"dispersion_center: {dispersion_center}"
    line2 = "" if not center_dispersion else f"center_dispersion: {center_dispersion}"
    line3 = "" if not width_dispersion else f"width_dispersion: {width_dispersion}"
    return dedent(
        f"""\
    irf:
        {irf_key}:
            type: {type}
            center: {center}
            width: {width}
            {line1}
            {line2}
            {line3}
    """
    )


BASE_MODEL_STR = f"""\
type: kinetic-spectrum
{get_dataset_block("[mc1]","j1","irf1")}
{get_initial_concentration_block("j1","[s1]","[j.1]")}
{get_megacomplex_block("mc1", "[k1]")}
{get_k_matrix_block("k1",[("(s1, s1)","kinetic.1")])}
{get_k_irf_block("irf1","spectral-gaussian","irf.center","irf.width","irf.dispcenter","[irf.centerdisp]")}
"""

SIM_MODEL_STR = f"""\
type: kinetic-spectrum
{get_dataset_block("[mc1]","j1","irf1",[('s1','sh1')])}
{get_initial_concentration_block("j1","[s1]","[j.1]")}
{get_megacomplex_block("mc1", "[k1]")}
{get_k_matrix_block("k1",[("(s1, s1)","kinetic.1")])}
{get_k_irf_block("irf1","spectral-gaussian","irf.center","irf.width","irf.dispcenter","[irf.centerdisp]")}
{get_shape_block("sh1", "one")}
"""


class SimpleIrfDispersion:

    print(BASE_MODEL_STR)
    print(SIM_MODEL_STR)
    print(BASE_MODEL_STR + SIM_MODEL_STR)

    model_ref = read_model_from_yaml(BASE_MODEL_STR)
    sim_ref = read_model_from_yaml(SIM_MODEL_STR)

    model = KineticSpectrumModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["j.1"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "kinetic.1",
                    }
                }
            },
            "irf": {
                "irf1": {
                    "type": "spectral-gaussian",
                    "center": "irf.center",
                    "width": "irf.width",
                    "dispersion_center": "irf.dispcenter",
                    "center_dispersion": ["irf.centerdisp"],
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
    assert model.markdown() == model_ref.markdown()  # Proves the two are equivalent
    sim_model = KineticSpectrumModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["j.1"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "kinetic.1",
                    }
                }
            },
            "irf": {
                "irf1": {
                    "type": "spectral-gaussian",
                    "center": "irf.center",
                    "width": "irf.width",
                    "dispersion_center": "irf.dispcenter",
                    "center_dispersion": ["irf.centerdisp"],
                },
            },
            "shape": {
                "sh1": {
                    "type": "one",
                },
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "irf": "irf1",
                    "megacomplex": ["mc1"],
                    "shape": {"s1": "sh1"},
                },
            },
        }
    )

    initial_parameters = ParameterGroup.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [["1", 0.5], {"non-negative": False}],
            "irf": [
                ["center", 0.3],
                ["width", 0.1],
                ["dispcenter", 400, {"vary": False}],
                ["centerdisp", 0.5],
            ],
        }
    )
    wanted_parameters = ParameterGroup.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [
                ["1", 0.5],
            ],
            "irf": [["center", 0.3], ["width", 0.1], ["dispcenter", 400], ["centerdisp", 0.5]],
        }
    )

    time_p1 = np.linspace(-1, 2, 50, endpoint=False)
    time_p2 = np.linspace(2, 5, 30, endpoint=False)
    time_p3 = np.geomspace(5, 10, num=20)
    time = np.concatenate([time_p1, time_p2, time_p3])
    spectral = np.arange(300, 500, 100)
    axis = {"time": time, "spectral": spectral}


class MultiIrfDispersion:
    model = KineticSpectrumModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["j.1"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "kinetic.1",
                    }
                }
            },
            "irf": {
                "irf1": {
                    "type": "spectral-multi-gaussian",
                    "center": ["irf.center"],
                    "width": ["irf.width"],
                    "dispersion_center": "irf.dispcenter",
                    "center_dispersion": ["irf.centerdisp1", "irf.centerdisp2"],
                    "width_dispersion": ["irf.widthdisp"],
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
    sim_model = KineticSpectrumModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["j.1"]},
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "kinetic.1",
                    }
                }
            },
            "irf": {
                "irf1": {
                    "type": "spectral-multi-gaussian",
                    "center": ["irf.center"],
                    "width": ["irf.width"],
                    "dispersion_center": "irf.dispcenter",
                    "center_dispersion": ["irf.centerdisp1", "irf.centerdisp2"],
                    "width_dispersion": ["irf.widthdisp"],
                },
            },
            "shape": {
                "sh1": {
                    "type": "one",
                },
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "irf": "irf1",
                    "megacomplex": ["mc1"],
                    "shape": {"s1": "sh1"},
                },
            },
        }
    )

    initial_parameters = ParameterGroup.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [["1", 0.5], {"non-negative": False}],
            "irf": [
                ["center", 0.3],
                ["width", 0.1],
                ["dispcenter", 400, {"vary": False}],
                ["centerdisp1", 0.01],
                ["centerdisp2", 0.001],
                ["widthdisp", 0.025],
            ],
        }
    )
    wanted_parameters = ParameterGroup.from_dict(
        {
            "j": [
                ["1", 1, {"vary": False, "non-negative": False}],
            ],
            "kinetic": [
                ["1", 0.5],
            ],
            "irf": [
                ["center", 0.3],
                ["width", 0.1],
                ["dispcenter", 400, {"vary": False}],
                ["centerdisp1", 0.01],
                ["centerdisp2", 0.001],
                ["widthdisp", 0.025],
            ],
        }
    )

    time = np.arange(-1, 5, 0.2)
    spectral = np.arange(300, 500, 100)
    axis = {"time": time, "spectral": spectral}


@pytest.mark.parametrize(
    "suite",
    [
        SimpleIrfDispersion,
        MultiIrfDispersion,
    ],
)
def test_spectral_irf(suite):

    model = suite.model
    print(model.validate())
    assert model.valid()

    sim_model = suite.sim_model
    print(sim_model.validate())
    assert sim_model.valid()

    wanted_parameters = suite.wanted_parameters
    print(sim_model.validate(wanted_parameters))
    print(wanted_parameters)
    assert sim_model.valid(wanted_parameters)

    initial_parameters = suite.initial_parameters
    print(model.validate(initial_parameters))
    assert model.valid(initial_parameters)

    print(model.markdown(wanted_parameters))

    dataset = sim_model.simulate("dataset1", wanted_parameters, suite.axis)

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

    print(resultdata)

    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, atol=1e-14)

    print(resultdata.fitted_data.isel(spectral=0).argmax())
    print(resultdata.fitted_data.isel(spectral=-1).argmax())
    assert (
        resultdata.fitted_data.isel(spectral=0).argmax()
        != resultdata.fitted_data.isel(spectral=-1).argmax()
    )

    assert "species_associated_spectra" in resultdata
    assert "decay_associated_spectra" in resultdata


if __name__ == "__main__":
    test_spectral_irf(SimpleIrfDispersion())
