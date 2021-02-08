import numpy as np
import pytest

from glotaran import read_model_from_yaml
from glotaran import read_parameters_from_yaml
from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme

MODEL_BASE = """\
type: kinetic-spectrum
dataset:
    dataset1:
        megacomplex: [mc1]
        initial_concentration: j1
        irf: irf1
        shape:
            s1: sh1
initial_concentration:
    j1:
        compartments: [s1]
        parameters: [j.1]
megacomplex:
    mc1:
        k_matrix: [k1]
k_matrix:
    k1:
        matrix:
            (s1, s1): kinetic.1
shape:
    sh1:
        type: one
"""
MODEL_SIMPLE_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        center_dispersion: [irf.center_dispersion]
"""
MODEL_MULTI_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center]
        width: [irf.width]
        dispersion_center: irf.dispersion_center
        center_dispersion: [irf.center_dispersion1, irf.center_dispersion2]
        width_dispersion: [irf.width_dispersion]
"""

PARAMETERS_BASE = """\
j:
    - ['1', 1, {'vary': False, 'non-negative': False}]
kinetic:
    - ['1', 0.5, {'non-negative': False}]
"""

PARAMETERS_SIMPLE_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ['center', 0.3]
    - ['width', 0.1]
    - ['dispersion_center', 400, {{'vary': False}}]
    - ['center_dispersion', 0.5]
"""

PARAMETERS_MULTI_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center", 0.3]
    - ["width", 0.1]
    - ["dispersion_center", 400, {{"vary": False}}]
    - ["center_dispersion1", 0.01]
    - ["center_dispersion2", 0.001]
    - ["width_dispersion", 0.025]
"""


class SimpleIrfDispersion:
    model = read_model_from_yaml(MODEL_SIMPLE_IRF_DISPERSION)
    parameters = read_parameters_from_yaml(PARAMETERS_SIMPLE_IRF_DISPERSION)
    time_p1 = np.linspace(-1, 2, 50, endpoint=False)
    time_p2 = np.linspace(2, 5, 30, endpoint=False)
    time_p3 = np.geomspace(5, 10, num=20)
    time = np.concatenate([time_p1, time_p2, time_p3])
    spectral = np.arange(300, 500, 100)
    axis = {"time": time, "spectral": spectral}


class MultiIrfDispersion:
    model = read_model_from_yaml(MODEL_MULTI_IRF_DISPERSION)
    parameters = read_parameters_from_yaml(PARAMETERS_MULTI_IRF_DISPERSION)
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

    parameters = suite.parameters
    print(model.validate(parameters))
    assert model.valid(parameters)

    dataset = model.simulate("dataset1", parameters, suite.axis)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

    data = {"dataset1": dataset}

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]

    # print(resultdata)

    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, atol=1e-14)

    irf_max_at_start = resultdata.fitted_data.isel(spectral=0).argmax(axis=0)
    irf_max_at_end = resultdata.fitted_data.isel(spectral=-1).argmax(axis=0)
    print(f" irf_max_at_start: {irf_max_at_start}\n irf_max_at_end: {irf_max_at_end}")
    # These should not be equal due to dispersion:
    assert irf_max_at_start != irf_max_at_end

    assert "species_associated_spectra" in resultdata
    assert "decay_associated_spectra" in resultdata
