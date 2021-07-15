import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import MultichannelMulticomponentDecay
from glotaran.analysis.test.models import OneCompartmentDecay
from glotaran.analysis.test.models import ThreeDatasetDecay
from glotaran.analysis.test.models import TwoCompartmentDecay
from glotaran.project import Scheme


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize(
    "method",
    [
        "TrustRegionReflection",
        "Dogbox",
        "Levenberg-Marquardt",
    ],
)
@pytest.mark.parametrize(
    "suite",
    [OneCompartmentDecay, TwoCompartmentDecay, ThreeDatasetDecay, MultichannelMulticomponentDecay],
)
def test_optimization(suite, index_dependent, grouped, weight, method):
    model = suite.model

    model.megacomplex["m1"].is_index_dependent = index_dependent

    print("Grouped:", grouped)
    print("Index dependent:", index_dependent)

    sim_model = suite.sim_model
    sim_model.megacomplex["m1"].is_index_dependent = index_dependent

    print(model.validate())
    assert model.valid()

    print(sim_model.validate())
    assert sim_model.valid()

    wanted_parameters = suite.wanted_parameters
    print(wanted_parameters)
    print(sim_model.validate(wanted_parameters))
    assert sim_model.valid(wanted_parameters)

    initial_parameters = suite.initial_parameters
    print(initial_parameters)
    print(model.validate(initial_parameters))
    assert model.valid(initial_parameters)
    assert (
        model.dataset["dataset1"].fill(model, initial_parameters).index_dependent()
        == index_dependent
    )

    nr_datasets = 3 if issubclass(suite, ThreeDatasetDecay) else 1
    data = {}
    for i in range(nr_datasets):
        global_axis = getattr(suite, "global_axis" if i == 0 else f"global_axis{i+1}")
        model_axis = getattr(suite, "model_axis" if i == 0 else f"model_axis{i+1}")

        dataset = simulate(
            sim_model,
            f"dataset{i+1}",
            wanted_parameters,
            {"global": global_axis, "model": model_axis},
        )
        print(f"Dataset {i+1}")
        print("=============")
        print(dataset)

        if hasattr(suite, "scale"):
            dataset["data"] /= suite.scale

        if weight:
            dataset["weight"] = xr.DataArray(
                np.ones_like(dataset.data) * 0.5, coords=dataset.data.coords
            )

        assert dataset.data.shape == (model_axis.size, global_axis.size)

        data[f"dataset{i+1}"] = dataset

    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data=data,
        maximum_number_function_evaluations=10,
        group=grouped,
        group_tolerance=0.1,
        optimization_method=method,
    )

    result = optimize(scheme)
    print(result.optimized_parameters)
    assert result.success
    optimized_scheme = result.get_scheme()
    assert result.optimized_parameters == optimized_scheme.parameters
    for dataset in optimized_scheme.data.values():
        assert "fitted_data" not in dataset
        if weight:
            assert "weight" in dataset
    for label, param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, wanted_parameters.get(label).value, rtol=1e-1)

    for i, dataset in enumerate(data.values()):
        resultdata = result.data[f"dataset{i+1}"]
        print(f"Result Data {i+1}")
        print("=================")
        print(resultdata)
        assert "residual" in resultdata
        assert "residual_left_singular_vectors" in resultdata
        assert "residual_right_singular_vectors" in resultdata
        assert "residual_singular_values" in resultdata
        assert np.array_equal(dataset.coords["model"], resultdata.coords["model"])
        assert np.array_equal(dataset.coords["global"], resultdata.coords["global"])
        assert dataset.data.shape == resultdata.data.shape
        print(dataset.data[0, 0], resultdata.data[0, 0])
        assert np.allclose(dataset.data, resultdata.data)
        if weight:
            assert "weight" in resultdata
            assert "weighted_residual" in resultdata
            assert "weighted_residual_left_singular_vectors" in resultdata
            assert "weighted_residual_right_singular_vectors" in resultdata
            assert "weighted_residual_singular_values" in resultdata
