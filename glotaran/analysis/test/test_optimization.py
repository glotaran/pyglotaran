import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.mock import MultichannelMulticomponentDecay
from glotaran.analysis.test.mock import OneCompartmentDecay
from glotaran.analysis.test.mock import TwoCompartmentDecay


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize(
    "suite", [OneCompartmentDecay, TwoCompartmentDecay, MultichannelMulticomponentDecay]
)
def test_optimization(suite, index_dependent, grouped, weight):
    model = suite.model

    model.grouped = lambda: grouped
    model.index_depended = lambda: index_dependent

    sim_model = suite.sim_model
    est_axis = suite.e_axis
    cal_axis = suite.c_axis

    print(model.validate())
    assert model.valid()

    print(sim_model.validate())
    assert sim_model.valid()

    wanted = suite.wanted
    print(wanted)
    print(sim_model.validate(wanted))
    assert sim_model.valid(wanted)

    initial = suite.initial
    print(initial)
    print(model.validate(initial))
    assert model.valid(initial)

    dataset = simulate(sim_model, "dataset1", wanted, {"e": est_axis, "c": cal_axis})
    print(dataset)

    if weight:
        dataset["weight"] = xr.DataArray(np.ones_like(dataset.data) * 0.5, coords=dataset.coords)

    assert dataset.data.shape == (cal_axis.size, est_axis.size)

    data = {"dataset1": dataset}
    scheme = Scheme(model=model, parameter=initial, data=data, nfev=10)

    result = optimize(scheme)
    print(result.optimized_parameter)
    print(result.data["dataset1"])

    for _, param in result.optimized_parameter.all():
        assert np.allclose(param.value, wanted.get(param.full_label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    print(resultdata)
    assert "residual" in resultdata
    assert "residual_left_singular_vectors" in resultdata
    assert "residual_right_singular_vectors" in resultdata
    assert "residual_singular_values" in resultdata
    assert np.array_equal(dataset.c, resultdata.c)
    assert np.array_equal(dataset.e, resultdata.e)
    assert dataset.data.shape == resultdata.data.shape
    print(dataset.data[0, 0], resultdata.data[0, 0])
    assert np.allclose(dataset.data, resultdata.data)

    if weight:
        assert "weight" in resultdata
        assert "weighted_residual" in resultdata
        assert "weighted_residual_left_singular_vectors" in resultdata
        assert "weighted_residual_right_singular_vectors" in resultdata
        assert "weighted_residual_singular_values" in resultdata


if __name__ == "__main__":
    test_optimization(OneCompartmentDecay, True, True, True)
    test_optimization(OneCompartmentDecay, False, False, False)
    test_optimization(OneCompartmentDecay, False, False, True)
