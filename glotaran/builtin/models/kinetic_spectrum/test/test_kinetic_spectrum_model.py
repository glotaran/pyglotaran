import numpy as np
import pytest

from glotaran import read_model_from_yaml
from glotaran import read_parameters_from_yaml
from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme

MODEL_ONE_COMPONENT_BASE = """\
type: kinetic-spectrum
dataset:
    dataset1: &dataset1
        megacomplex: [mc1]
        initial_concentration: j1
initial_concentration:
    j1:
        compartments: [s1]
        parameters: ["1"]
megacomplex:
    mc1:
        k_matrix: [k1]
k_matrix:
    k1:
        matrix:
            (s1, s1): "2"
"""
MODEL_ONE_COMPONENT = MODEL_ONE_COMPONENT_BASE

MODEL_SIM_ONE_COMPONENT = f"""\
{MODEL_ONE_COMPONENT_BASE}
dataset:
    dataset1:
        <<: *dataset1
        shape:
            s1: sh1
shape:
    sh1:
        type: one
"""

PARAMETERS_BASE = """\
- [1, {"vary": False, "non-negative": False}]
"""

PARAMETERS_ONE_COMPONENT_INITIAL = f"""\
{PARAMETERS_BASE}
- 101e-4
"""

PARAMETERS_ONE_COMPONENT_WANTED = f"""\
{PARAMETERS_BASE}
- 101e-3
"""


MODEL_ONE_COMPONENT_ONE_CHANNEL_GASSIAN = f"""\
{MODEL_ONE_COMPONENT_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: "3"
        width: "4"
dataset:
    dataset1:
        <<: *dataset1
        irf: irf1
"""

MODEL_SIM_ONE_COMPONENT_ONE_CHANNEL_GASSIAN = f"""\
{MODEL_ONE_COMPONENT_ONE_CHANNEL_GASSIAN}
dataset:
    dataset1:
        <<: *dataset1
        irf: irf1
        shape:
            s1: sh1
shape:
    sh1:
        type: one
"""

PARAMETERS_ONE_COMPONENT_ONE_CHANNEL_GASSIAN_INITIAL = f"""\
{PARAMETERS_BASE}
- 101e-4
- 0.1
- 1
"""

PARAMETERS_ONE_COMPONENT_ONE_CHANNEL_GASSIAN_WANTED = f"""\
{PARAMETERS_BASE}
- 101e-3
- 0.3
- 2
"""


class OneComponentOneChannel:
    model = read_model_from_yaml(MODEL_ONE_COMPONENT)
    sim_model = read_model_from_yaml(MODEL_SIM_ONE_COMPONENT)
    initial_parameters = read_parameters_from_yaml(PARAMETERS_ONE_COMPONENT_INITIAL)
    wanted_parameters = read_parameters_from_yaml(PARAMETERS_ONE_COMPONENT_WANTED)
    time = np.asarray(np.arange(0, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}


class OneComponentOneChannelGaussianIrf:
    model = read_model_from_yaml(MODEL_ONE_COMPONENT_ONE_CHANNEL_GASSIAN)
    sim_model = read_model_from_yaml(MODEL_SIM_ONE_COMPONENT_ONE_CHANNEL_GASSIAN)
    initial_parameters = read_parameters_from_yaml(
        PARAMETERS_ONE_COMPONENT_ONE_CHANNEL_GASSIAN_INITIAL
    )
    wanted_parameters = read_parameters_from_yaml(
        PARAMETERS_ONE_COMPONENT_ONE_CHANNEL_GASSIAN_WANTED
    )
    time = np.asarray(np.arange(-10, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}


MODEL_3C_BASE = """\
type: kinetic-spectrum
dataset:
    dataset1: &dataset1
        megacomplex: [mc1]
        initial_concentration: j1
        irf: irf1
megacomplex:
    mc1:
        k_matrix: [k1]
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center]
        width: [irf.width]
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

MODEL_3C_SIM_BASE = """\
dataset:
    dataset1:
        <<: *dataset1
        shape:
            s1: sh1
            s2: sh2
            s3: sh3
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

MODEL_SIM_3C_PARALLEL = f"""\
{MODEL_3C_PARALLEL}
{MODEL_3C_SIM_BASE}
"""

MODEL_SIM_3C_SEQUENTIAL = f"""\
{MODEL_3C_SEQUENTIAL}
{MODEL_3C_SIM_BASE}
"""

PARAMETERS_3C_BASE_PARALLEL = """\
irf:
    - ["center", 1.3]
    - ["width", 7.8]
j:
    - ["1", 1, {"vary": False, "non-negative": False}]
"""

PARAMETERS_3C_SIM_PARALLEL = f"""\
kinetic:
    - ["1", 301e-3]
    - ["2", 502e-4]
    - ["3", 705e-5]
shapes:
    amps: [7, 3, 30, {{"vary": False}}]
    locs: [620, 670, 720, {{"vary": False}}]
    width: [10, 30, 50, {{"vary": False}}]
{PARAMETERS_3C_BASE_PARALLEL}
"""

PARAMETERS_3C_INITIAL_PARALLEL = f"""\
kinetic:
    - ["1", 300e-3]
    - ["2", 500e-4]
    - ["3", 700e-5]
{PARAMETERS_3C_BASE_PARALLEL}
"""

PARAMETERS_3C_BASE_SEQUENTIAL = """\
irf:
    - ["center", 1.3]
    - ["width", 7.8]
j:
    - ["1", 1, {"vary": False, "non-negative": False}]
    - ["0", 0, {"vary": False, "non-negative": False}]
"""

PARAMETERS_3C_SIM_SEQUENTIAL = f"""\
kinetic:
    - ["1", 501e-3]
    - ["2", 202e-4]
    - ["3", 105e-5]
shapes:
    amps: [3, 1, 5, {{"vary": False}}]
    locs: [620, 670, 720, {{"vary": False}}]
    width: [10, 30, 50, {{"vary": False}}]
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
    model = read_model_from_yaml(MODEL_3C_PARALLEL)
    sim_model = read_model_from_yaml(MODEL_SIM_3C_PARALLEL)
    initial_parameters = read_parameters_from_yaml(PARAMETERS_3C_INITIAL_PARALLEL)
    wanted_parameters = read_parameters_from_yaml(PARAMETERS_3C_SIM_PARALLEL)
    time = np.arange(-10, 100, 1.5)
    spectral = np.arange(600, 750, 10)
    axis = {"time": time, "spectral": spectral}


class ThreeComponentSequential:
    model = read_model_from_yaml(MODEL_3C_SEQUENTIAL)
    sim_model = read_model_from_yaml(MODEL_SIM_3C_SEQUENTIAL)
    initial_parameters = read_parameters_from_yaml(PARAMETERS_3C_INITIAL_SEQUENTIAL)
    wanted_parameters = read_parameters_from_yaml(PARAMETERS_3C_SIM_SEQUENTIAL)
    time = np.asarray(np.arange(-10, 50, 1.0))
    spectral = np.arange(600, 750, 5.0)
    axis = {"time": time, "spectral": spectral}


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
        non_negative_least_squares=nnls,
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
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)

    assert "species_associated_spectra" in resultdata
    assert "decay_associated_spectra" in resultdata


if __name__ == "__main__":
    test_kinetic_model(ThreeComponentParallel(), True)
    test_kinetic_model(ThreeComponentParallel(), False)
    test_kinetic_model(ThreeComponentSequential(), True)
    test_kinetic_model(ThreeComponentSequential(), False)
