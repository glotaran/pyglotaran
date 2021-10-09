import numpy as np
import pytest

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.project import Scheme

MODEL_3C_BASE = """\
dataset:
    dataset1:
        megacomplex: [mc1]
        global_megacomplex: [mc2]
        initial_concentration: j1
        irf: irf1
    dataset2:
        megacomplex: [mc2]
        global_megacomplex: [mc1]
        initial_concentration: j1
        irf: irf1
    dataset3:
        megacomplex: [mc1]
        initial_concentration: j1
        irf: irf1
    dataset4:
        megacomplex: [mc2]
megacomplex:
    mc1:
        type: decay
        k_matrix: [k1]
    mc2:
        type: spectral
        shape:
            s1: sh1
            s2: sh2
            s3: sh3
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center]
        width: [irf.width]
shape:
    sh1:
        type: gaussian
        amplitude: shapes.amps.1
        location: shapes.locs.1
        width: shapes.width.1
    sh2:
        type: gaussian
        amplitude: shapes.amps.2
        location: shapes.locs.2
        width: shapes.width.2
    sh3:
        type: gaussian
        amplitude: shapes.amps.3
        location: shapes.locs.3
        width: shapes.width.3
"""

MODEL_3C_PARALLEL = f"""\
{MODEL_3C_BASE}
initial_concentration:
    j1:
        compartments: [s1, s2, s3]
        parameters: [j.1, j.1, j.1]
k_matrix:
    k1:
        matrix:
            (s1, s1): "kinetic.1"
            (s2, s2): "kinetic.2"
            (s3, s3): "kinetic.3"
"""

MODEL_3C_SEQUENTIAL = f"""\
{MODEL_3C_BASE}
initial_concentration:
    j1:
        compartments: [s1, s2, s3]
        parameters: [j.1, j.0, j.0]
k_matrix:
    k1:
        matrix:
            (s2, s1): "kinetic.1"
            (s3, s2): "kinetic.2"
            (s3, s3): "kinetic.3"
"""

PARAMETERS_3C_BASE = """\
irf:
    - ["center", 1.3]
    - ["width", 7.8]
j:
    - ["1", 1, {"vary": False, "non-negative": False}]
    - ["0", 0, {"vary": False, "non-negative": False}]
"""

PARAMETERS_3C_BASE_PARALLEL = f"""\
{PARAMETERS_3C_BASE}
shapes:
    amps: [7, 3, 30, {{"vary": False}}]
    locs: [620, 670, 720, {{"vary": False}}]
    width: [10, 30, 50, {{"vary": False}}]
"""

PARAMETERS_3C_BASE_SEQUENTIAL = f"""\
{PARAMETERS_3C_BASE}
shapes:
    amps:
        - 9
        - 7
        - 5
        - {{"vary": True, min: 0, max: 10}}
    locs: [610, 670, 730, {{"vary": False}}]
    width: [15, 25, 10, {{"vary": False}}]
"""

PARAMETERS_3C_PARALLEL_WANTED = f"""\
kinetic:
    - ["1", 301e-3]
    - ["2", 502e-4]
    - ["3", 705e-5]
{PARAMETERS_3C_BASE_PARALLEL}
"""

PARAMETERS_3C_INITIAL_PARALLEL = f"""\
kinetic:
    - ["1", 300e-3, {{non-negative: true}}]
    - ["2", 500e-4, {{non-negative: true}}]
    - ["3", 700e-5, {{non-negative: true}}]
{PARAMETERS_3C_BASE_PARALLEL}
"""

PARAMETERS_3C_SIM_SEQUENTIAL = f"""\
kinetic:
    - ["1", 501e-3, {{non-negative: true}}]
    - ["2", 202e-4, {{non-negative: true}}]
    - ["3", 105e-5, {{non-negative: true}}]
{PARAMETERS_3C_BASE_SEQUENTIAL}
"""

PARAMETERS_3C_INITIAL_SEQUENTIAL = f"""\
kinetic:
    - ["1", 500e-3]
    - ["2", 200e-4]
    - ["3", 100e-5]
    - {{"non-negative": True}}
{PARAMETERS_3C_BASE_SEQUENTIAL}
"""


class ThreeComponentParallel:
    model = load_model(MODEL_3C_PARALLEL, format_name="yml_str")
    initial_parameters = load_parameters(PARAMETERS_3C_INITIAL_PARALLEL, format_name="yml_str")
    wanted_parameters = load_parameters(PARAMETERS_3C_PARALLEL_WANTED, format_name="yml_str")
    time = np.arange(-10, 100, 1.5)
    spectral = np.arange(600, 750, 10)
    axis = {"time": time, "spectral": spectral}


class ThreeComponentSequential:
    model = load_model(MODEL_3C_SEQUENTIAL, format_name="yml_str")
    initial_parameters = load_parameters(PARAMETERS_3C_INITIAL_SEQUENTIAL, format_name="yml_str")
    wanted_parameters = load_parameters(PARAMETERS_3C_SIM_SEQUENTIAL, format_name="yml_str")
    time = np.arange(-10, 50, 1.0)
    spectral = np.arange(600, 750, 5.0)
    axis = {"time": time, "spectral": spectral}


@pytest.mark.parametrize(
    "suite",
    [
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

    print(model.markdown(wanted_parameters))

    dataset = simulate(model, "dataset1", wanted_parameters, suite.axis)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

    data = {f"dataset{i}": dataset for i in range(1, 5)}

    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for label, param in result.optimized_parameters.all():
        print(label, param.value, wanted_parameters.get(label).value)
        assert np.allclose(param.value, wanted_parameters.get(label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]

    print(resultdata)

    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)
