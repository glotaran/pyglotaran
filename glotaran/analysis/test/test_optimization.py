import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.mock import DecayModel
from glotaran.analysis.test.mock import MultichannelMulticomponentDecay
from glotaran.analysis.test.mock import OneCompartmentDecay
from glotaran.analysis.test.mock import ThreeDatasetDecay
from glotaran.analysis.test.mock import TwoCompartmentDecay


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize(
    "method",
    [
        "TrustRegionReflection",
        "Dogbox",
        "LevenbergMarquart",
    ],
)
@pytest.mark.parametrize(
    "suite",
    [OneCompartmentDecay, TwoCompartmentDecay, ThreeDatasetDecay, MultichannelMulticomponentDecay],
)
def test_optimization(suite, index_dependent, grouped, weight, method):
    if issubclass(suite, ThreeDatasetDecay) and method == "LevenbergMarquart":
        # LM does not support bounds
        return
    model = suite.model

    model.is_grouped = grouped
    model.is_index_dependent = index_dependent
    print("Weighted:", weight)
    print("Grouped:", grouped)
    print("Index dependent:", index_dependent)

    assert model.grouped() == grouped
    assert model.index_dependent() == index_dependent

    sim_model = suite.sim_model
    sim_model.is_grouped = grouped
    sim_model.is_index_dependent = index_dependent

    print(model.validate())
    assert model.valid()

    print(sim_model.validate())
    assert sim_model.valid()

    wanted = suite.wanted
    print(wanted)
    print(sim_model.validate(wanted))
    assert sim_model.valid(wanted)

    initial = suite.initial.copy()
    print(initial)
    print(model.validate(initial))
    assert model.valid(initial)

    nr_datasets = 3 if issubclass(suite, ThreeDatasetDecay) else 1
    data = {}
    for i in range(nr_datasets):
        e_axis = getattr(suite, "e_axis" if i == 0 else f"e_axis{i+1}")
        c_axis = getattr(suite, "c_axis" if i == 0 else f"c_axis{i+1}")

        dataset = simulate(sim_model, f"dataset{i+1}", wanted, {"e": e_axis, "c": c_axis})
        print(f"Dataset {i+1}")
        print("=============")
        print(dataset)

        if weight:
            dataset["weight"] = xr.DataArray(
                np.ones_like(dataset.data) * 0.5, coords=dataset.coords
            )

        assert dataset.data.shape == (c_axis.size, e_axis.size)

        data[f"dataset{i+1}"] = dataset

    scheme = Scheme(
        model=model,
        parameter=initial,
        data=data,
        nfev=20,
        group_tolerance=0.1,
        optimization_method=method,
    )

    result = optimize(scheme)
    print(result.optimized_parameter)

    for label, param in result.optimized_parameter.all():
        if param.vary:
            assert np.allclose(param.value, wanted.get(label).value, rtol=1e-1)

    for i, dataset in enumerate(data.values()):
        resultdata = result.data[f"dataset{i+1}"]
        print(f"Result Data {i+1}")
        print("=================")
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

    assert callable(model.additional_penalty_function)
    assert model.additional_penalty_function_called

    if isinstance(model, DecayModel):
        assert callable(model.constrain_matrix_function)
        assert model.constrain_matrix_function_called
        assert callable(model.retrieve_clp_function)
        assert model.retrieve_clp_function_called
    else:
        assert not model.constrain_matrix_function_called
        assert not model.retrieve_clp_function_called


if __name__ == "__main__":
    test_optimization(OneCompartmentDecay, True, True, True)
    test_optimization(OneCompartmentDecay, False, False, False)
    test_optimization(OneCompartmentDecay, False, False, True)
