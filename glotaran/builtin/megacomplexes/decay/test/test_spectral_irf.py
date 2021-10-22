import warnings
from copy import deepcopy
from textwrap import dedent

import numpy as np
import pytest

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.project import Scheme

MODEL_BASE = """\
default_megacomplex: decay
dataset:
    dataset1:
        megacomplex: [mc1]
        initial_concentration: j1
        irf: irf1
initial_concentration:
    j1:
        compartments: [s1]
        parameters: [j.1]
megacomplex:
    mc1:
        k_matrix: [k1]
    mc2:
        type: spectral
        shape:
            s1: sh1
k_matrix:
    k1:
        matrix:
            (s1, s1): kinetic.1
shape:
    sh1:
        type: one
"""
MODEL_NO_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
"""
MODEL_SIMPLE_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        center_dispersion_coefficients: [irf.cdc1]
"""
MODEL_MULTI_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center]
        width: [irf.width]
        dispersion_center: irf.dispersion_center
        center_dispersion_coefficients: [irf.cdc1, irf.cdc2]
        width_dispersion_coefficients: [irf.wdc1]
"""

MODEL_MULTIPULSE_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center1, irf.center2]
        width: [irf.width]
        dispersion_center: irf.dispersion_center
        center_dispersion_coefficients: [irf.cdc1, irf.cdc2, irf.cdc3]
"""

PARAMETERS_BASE = """\
j:
    - ['1', 1, {'vary': False, 'non-negative': False}]
kinetic:
    - ['1', 0.5, {'non-negative': False}]
"""

PARAMETERS_NO_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ['center', 0.3]
    - ['width', 0.1]
"""

PARAMETERS_SIMPLE_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ['center', 0.3]
    - ['width', 0.1]
    - ['dispersion_center', 400, {{'vary': False}}]
    - ['cdc1', 0.5]
"""

# What is this?
PARAMETERS_MULTI_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center", 0.3]
    - ["width", 0.1]
    - ["dispersion_center", 400, {{"vary": False}}]
    - ["cdc1", 0.1]
    - ["cdc2", 0.01]
    - ["wdc1", 0.025]
"""

PARAMETERS_MULTIPULSE_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center1", 0.3]
    - ["center2", 0.4]
    - ['width', 0.1]
    - ['dispersion_center', 400, {{'vary': False}}]
    - ["cdc1", 0.5]
    - ["cdc2", 0.1]
    - ["cdc3", -0.01]
"""


def _time_axis():
    time_p1 = np.linspace(-1, 1, 20, endpoint=False)
    time_p2 = np.linspace(1, 2, 10, endpoint=False)
    time_p3 = np.geomspace(2, 20, num=20)
    return np.array(np.concatenate([time_p1, time_p2, time_p3]))


def _spectral_axis():
    return np.linspace(300, 500, 3)


def _calculate_irf_position(
    index, center, dispersion_center=None, center_dispersion_coefficients=None
):
    if center_dispersion_coefficients is None:
        center_dispersion_coefficients = []
    if dispersion_center is not None:
        distance = (index - dispersion_center) / 100
        for i, coefficient in enumerate(center_dispersion_coefficients):
            center += coefficient * np.power(distance, i + 1)
    return center


class NoIrfDispersion:
    model = load_model(MODEL_NO_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_NO_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class SimpleIrfDispersion:
    model = load_model(MODEL_SIMPLE_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_SIMPLE_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class MultiIrfDispersion:
    model = load_model(MODEL_MULTI_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTI_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class MultiCenterIrfDispersion:
    model = load_model(MODEL_MULTIPULSE_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTIPULSE_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


@pytest.mark.parametrize(
    "suite",
    [
        NoIrfDispersion,
        SimpleIrfDispersion,
        MultiIrfDispersion,
        MultiCenterIrfDispersion,
    ],
)
def test_spectral_irf(suite):

    model = suite.model
    assert model.valid(), model.validate()

    parameters = suite.parameters
    assert model.valid(parameters), model.validate(parameters)

    sim_model = deepcopy(model)
    sim_model.dataset["dataset1"].global_megacomplex = ["mc2"]
    dataset = simulate(sim_model, "dataset1", parameters, suite.axis)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

    data = {"dataset1": dataset}

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(label).value), dedent(
            f"""
            Error in {suite.__name__} comparing {param.full_label},
            - diff={param.value-parameters.get(label).value}
            """
        )

    resultdata = result.data["dataset1"]

    # print(resultdata)
    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    # assert np.allclose(dataset.data, resultdata.fitted_data, atol=1e-14)

    fit_data_max_at_start = resultdata.fitted_data.isel(spectral=0).argmax(axis=0)
    fit_data_max_at_end = resultdata.fitted_data.isel(spectral=-1).argmax(axis=0)

    if suite is NoIrfDispersion:
        assert "center_dispersion_1" not in resultdata
        assert fit_data_max_at_start == fit_data_max_at_end
    else:
        assert "center_dispersion_1" in resultdata
        assert fit_data_max_at_start != fit_data_max_at_end
        if abs(fit_data_max_at_start - fit_data_max_at_end) < 3:
            warnings.warn(
                dedent(
                    """
                    Bad test, one of the following could be the case:
                    - dispersion too small
                    - spectral window to small
                    - time resolution (around the maximum of the IRF) too low"
                    """
                )
            )

        for x in suite.axis["spectral"]:
            # calculated irf location
            model_irf_center = suite.model.irf["irf1"].center
            model_dispersion_center = suite.model.irf["irf1"].dispersion_center
            model_center_dispersion_coefficients = suite.model.irf[
                "irf1"
            ].center_dispersion_coefficients
            calc_irf_location_at_x = _calculate_irf_position(
                x, model_irf_center, model_dispersion_center, model_center_dispersion_coefficients
            )
            # fitted irf location
            fitted_irf_loc_at_x = resultdata["irf_center_location"].sel(spectral=x)
            assert np.allclose(calc_irf_location_at_x, fitted_irf_loc_at_x.values), dedent(
                f"""
                Error in {suite.__name__} comparing irf_center_location,
                - diff={calc_irf_location_at_x-fitted_irf_loc_at_x.values}
                """
            )

    assert "species_associated_spectra" in resultdata
    assert "decay_associated_spectra" in resultdata
    assert "irf_center" in resultdata
