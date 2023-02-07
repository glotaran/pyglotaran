import numpy as np
import pytest
import xarray as xr

from glotaran.optimization.optimize import optimize
from glotaran.optimization.test.models import SimpleTestModel
from glotaran.optimization.test.suites import FullModel
from glotaran.optimization.test.suites import MultichannelMulticomponentDecay
from glotaran.optimization.test.suites import OneCompartmentDecay
from glotaran.optimization.test.suites import ThreeDatasetDecay
from glotaran.optimization.test.suites import TwoCompartmentDecay
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


@pytest.mark.parametrize("is_index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
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
def test_optimization(suite, is_index_dependent, link_clp, weight, method):
    model = suite.model

    model.megacomplex["m1"].is_index_dependent = is_index_dependent

    print("Link CLP:", link_clp)  # T201
    print("Index dependent:", is_index_dependent)  # T201

    sim_model = suite.sim_model
    sim_model.megacomplex["m1"].is_index_dependent = is_index_dependent

    print(model.validate())  # T201
    assert model.valid()

    print(sim_model.validate())  # T201
    assert sim_model.valid()

    wanted_parameters = suite.wanted_parameters
    print(wanted_parameters)  # T201
    print(sim_model.validate(wanted_parameters))  # T201
    assert sim_model.valid(wanted_parameters)

    initial_parameters = suite.initial_parameters
    print(initial_parameters)  # T201
    print(model.validate(initial_parameters))  # T201
    assert model.valid(initial_parameters)

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
        print(f"Dataset {i+1}")  # T201
        print("=============")  # T201
        print(dataset.data)  # T201

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
        clp_link_tolerance=0.1,
        optimization_method=method,
    )

    model.dataset_groups["default"].link_clp = link_clp

    result = optimize(scheme, raise_exception=True)
    print(result.optimized_parameters)  # T201
    print(result.data["dataset1"].fitted_data)  # T201
    assert result.success
    optimized_scheme = result.get_scheme()
    assert result.optimized_parameters != initial_parameters
    assert result.optimized_parameters == optimized_scheme.parameters
    for dataset in optimized_scheme.data.values():
        assert "fitted_data" not in dataset
        if weight:
            assert "weight" in dataset
    for param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, wanted_parameters.get(param.label).value, rtol=1e-1)

    for i, dataset in enumerate(data.values()):
        resultdata = result.data[f"dataset{i+1}"]
        print(f"Result Data {i+1}")  # T201k
        print("=================")  # T201
        print(resultdata)  # T201
        assert "matrix" in resultdata
        assert len(resultdata.matrix.shape) == (3 if is_index_dependent else 2)
        assert "residual" in resultdata
        assert "residual_left_singular_vectors" in resultdata
        assert "residual_right_singular_vectors" in resultdata
        assert "residual_singular_values" in resultdata
        assert np.array_equal(dataset.coords["model"], resultdata.coords["model"])
        assert np.array_equal(dataset.coords["global"], resultdata.coords["global"])
        assert dataset.data.shape == resultdata.data.shape
        print(dataset.data[0, 0], resultdata.data[0, 0])  # T201
        assert np.allclose(dataset.data, resultdata.data)
        if weight:
            assert "weight" in resultdata
            assert "weighted_residual" in resultdata
            assert "weighted_residual_left_singular_vectors" in resultdata
            assert "weighted_residual_right_singular_vectors" in resultdata
            assert "weighted_residual_singular_values" in resultdata


@pytest.mark.parametrize("index_dependent", [True, False])
def test_optimization_full_model(index_dependent):
    model = FullModel.model
    model.megacomplex["m1"].is_index_dependent = index_dependent

    print(model.validate())  # T201
    assert model.valid()

    parameters = FullModel.parameters
    assert model.valid(parameters)

    dataset = simulate(model, "dataset1", parameters, FullModel.coordinates)

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data={"dataset1": dataset},
        maximum_number_function_evaluations=10,
    )

    result = optimize(scheme, raise_exception=True)
    assert result.success
    optimized_scheme = result.get_scheme()
    assert result.optimized_parameters == optimized_scheme.parameters

    result_data = result.data["dataset1"]
    assert "fitted_data" in result_data
    for param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, parameters.get(param.label).value, rtol=1e-1)

    clp = result_data.clp
    print(clp)  # T201
    assert clp.shape == (4, 4)
    assert all(np.isclose(1.0, c) for c in np.diagonal(clp))


@pytest.mark.parametrize("model_weight", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_result_data(model_weight: bool, index_dependent: bool):
    global_axis = [1, 5, 6]
    model_axis = [5, 7, 9, 12]

    data = xr.DataArray(
        np.ones((4, 3)), coords=[("model", model_axis), ("global", global_axis)]
    ).to_dataset(name="data")

    model_dict = {
        "megacomplex": {"m1": {"type": "simple-test-mc", "is_index_dependent": index_dependent}},
        "dataset": {"dataset1": {"megacomplex": ["m1"]}},
    }

    if model_weight:
        model_dict["weights"] = [{"datasets": ["dataset1"], "value": 0.5}]
    else:
        data["weight"] = xr.ones_like(data.data) * 0.5

    model = SimpleTestModel(**model_dict)
    assert model.valid()
    parameters = Parameters.from_list([1])

    scheme = Scheme(model, parameters, {"dataset1": data}, maximum_number_function_evaluations=1)
    result = optimize(scheme, raise_exception=True)
    result_data = result.data["dataset1"]
    wanted = [
        ("data", ("model", "global")),
        ("data_left_singular_vectors", ("model", "left_singular_value_index")),
        ("data_singular_values", ("singular_value_index",)),
        ("data_right_singular_vectors", ("global", "right_singular_value_index")),
        ("clp", ("global", "clp_label")),
        ("weight", ("model", "global")),
        ("weighted_residual", ("model", "global")),
        ("residual", ("model", "global")),
        ("weighted_residual_left_singular_vectors", ("model", "left_singular_value_index")),
        ("weighted_residual_singular_values", ("singular_value_index",)),
        ("weighted_residual_right_singular_vectors", ("global", "right_singular_value_index")),
        ("residual_left_singular_vectors", ("model", "left_singular_value_index")),
        ("residual_singular_values", ("singular_value_index",)),
        ("residual_right_singular_vectors", ("global", "right_singular_value_index")),
        ("fitted_data", ("model", "global")),
    ]

    for label, dims in wanted:
        print("Check label", label)
        assert label in result_data
        print("Check dims", result_data[label].dims, dims)
        assert result_data[label].dims == dims

    assert "matrix" in result_data

    if index_dependent:
        assert result_data.matrix.dims == ("global", "model", "clp_label")
    else:
        assert result_data.matrix.dims == ("model", "clp_label")
