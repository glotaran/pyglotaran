import collections

import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimization_group import OptimizationGroup
from glotaran.analysis.optimization_group_calculator_linked import (
    OptimizationGroupCalculatorLinked,
)
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import FullModel
from glotaran.analysis.test.models import MultichannelMulticomponentDecay as suite
from glotaran.analysis.test.models import SimpleTestModel
from glotaran.analysis.util import CalculatedMatrix
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


@pytest.fixture(
    scope="module", params=[[True, True], [True, False], [False, True], [False, False]]
)
def optimization_group(request) -> OptimizationGroup:
    model = suite.model
    model.megacomplex["m1"].is_index_dependent = request.param[1]
    model.is_index_dependent = request.param[1]
    model.dataset_group_models["default"].link_clp = request.param[0]

    dataset = simulate(
        suite.sim_model,
        "dataset1",
        suite.wanted_parameters,
        {"global": suite.global_axis, "model": suite.model_axis},
    )
    scheme = Scheme(model=model, parameters=suite.initial_parameters, data={"dataset1": dataset})

    return OptimizationGroup(scheme, model.get_dataset_groups()["default"])


def test_optimization_group_bag(optimization_group: OptimizationGroup):

    if isinstance(optimization_group._calculator, OptimizationGroupCalculatorLinked):
        bag = optimization_group._calculator.bag
        assert isinstance(bag, collections.deque)
        assert len(bag) == suite.global_axis.size
        assert optimization_group._calculator.groups == {"dataset1": ["dataset1"]}


def test_optimization_group_matrices(optimization_group: OptimizationGroup):
    optimization_group._calculator.calculate_matrices()

    if isinstance(optimization_group._calculator, OptimizationGroupCalculatorLinked):
        if optimization_group.model.is_index_dependent:
            assert all(
                isinstance(m, CalculatedMatrix) for m in optimization_group.reduced_matrices
            )
            assert len(optimization_group.reduced_matrices) == suite.global_axis.size
        else:
            assert "dataset1" in optimization_group.reduced_matrices
            assert isinstance(optimization_group.reduced_matrices["dataset1"], CalculatedMatrix)
    else:
        if optimization_group.model.is_index_dependent:
            assert isinstance(optimization_group.reduced_matrices, dict)
            assert isinstance(optimization_group.reduced_matrices["dataset1"], list)
            assert all(
                isinstance(m, CalculatedMatrix)
                for m in optimization_group.reduced_matrices["dataset1"]
            )
        else:
            assert isinstance(optimization_group.reduced_matrices["dataset1"], CalculatedMatrix)

        assert isinstance(optimization_group.matrices, dict)
        assert "dataset1" in optimization_group.reduced_matrices


def test_optimization_group_residuals(optimization_group: OptimizationGroup):
    optimization_group._calculator.calculate_residual()
    if isinstance(optimization_group._calculator, OptimizationGroupCalculatorLinked):
        assert isinstance(optimization_group.residuals, list)
        assert all(isinstance(r, np.ndarray) for r in optimization_group.residuals)
        assert len(optimization_group.residuals) == suite.global_axis.size
    else:
        assert isinstance(optimization_group.residuals, dict)
        assert "dataset1" in optimization_group.residuals
        assert all(isinstance(r, np.ndarray) for r in optimization_group.residuals["dataset1"])
        assert len(optimization_group.residuals["dataset1"]) == suite.global_axis.size


def test_optimization_group_result_data(optimization_group: OptimizationGroup):

    data = optimization_group.create_result_data()
    label = "dataset1"

    assert label in data

    dataset = data[label]
    dataset_model = optimization_group.dataset_models[label]

    assert "clp_label" in dataset.coords
    assert np.array_equal(dataset.clp_label, ["s1", "s2", "s3", "s4"])

    assert dataset_model.get_global_dimension() in dataset.coords
    assert np.array_equal(dataset.coords[dataset_model.get_global_dimension()], suite.global_axis)

    assert dataset_model.get_model_dimension() in dataset.coords
    assert np.array_equal(dataset.coords[dataset_model.get_model_dimension()], suite.model_axis)

    assert "matrix" in dataset
    matrix = dataset.matrix
    if optimization_group.model.is_index_dependent:
        assert len(matrix.shape) == 3
        assert matrix.shape[0] == suite.global_axis.size
        assert matrix.shape[1] == suite.model_axis.size
        assert matrix.shape[2] == 4
    else:
        assert len(matrix.shape) == 2
        assert matrix.shape[0] == suite.model_axis.size
        assert matrix.shape[1] == 4

    assert "clp" in dataset
    clp = dataset.clp
    assert len(clp.shape) == 2
    assert clp.shape[0] == suite.global_axis.size
    assert clp.shape[1] == 4

    assert "weighted_residual" in dataset
    assert dataset.data.shape == dataset.weighted_residual.shape

    assert "residual" in dataset
    assert dataset.data.shape == dataset.residual.shape

    assert "residual_singular_values" in dataset
    assert "weighted_residual_singular_values" in dataset


def test_prepare_data():
    model_dict = {
        "megacomplex": {"m1": {"is_index_dependent": False}},
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
            },
        },
        "weights": [
            {
                "datasets": ["dataset1"],
                "global_interval": (np.inf, 200),
                "model_interval": (4, 8),
                "value": 0.5,
            },
        ],
    }
    model = SimpleTestModel.from_dict(model_dict)
    print(model.validate())
    assert model.valid()

    parameters = ParameterGroup.from_list([])

    global_axis = np.asarray(range(50, 300))
    model_axis = np.asarray(range(15))

    dataset = xr.DataArray(
        np.ones((global_axis.size, model_axis.size)),
        coords={"global": global_axis, "model": model_axis},
        dims=("global", "model"),
    )

    scheme = Scheme(model, parameters, {"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])

    data = optimization_group.data["dataset1"]
    print(data)
    assert "data" in data
    assert "weight" in data

    assert data.data.shape == (model_axis.size, global_axis.size)
    assert data.data.shape == data.weight.shape
    assert np.all(data.weight.sel({"global": slice(0, 200), "model": slice(4, 8)}).values == 0.5)
    assert np.all(data.weight.sel(model=slice(0, 3)).values == 1)

    model_dict["weights"].append(
        {
            "datasets": ["dataset1"],
            "value": 0.2,
        }
    )
    model = SimpleTestModel.from_dict(model_dict)
    print(model.validate())
    assert model.valid()

    scheme = Scheme(model, parameters, {"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])

    data = optimization_group.data["dataset1"]
    assert np.all(
        data.weight.sel({"global": slice(0, 200), "model": slice(4, 8)}).values == 0.5 * 0.2
    )
    assert np.all(data.weight.sel(model=slice(0, 3)).values == 0.2)

    with pytest.warns(
        UserWarning,
        match="Ignoring model weight for dataset 'dataset1'"
        " because weight is already supplied by dataset.",
    ):
        OptimizationGroup(
            Scheme(model, parameters, {"dataset1": data}), model.get_dataset_groups()["default"]
        )


def test_full_model_problem():
    dataset = simulate(FullModel.model, "dataset1", FullModel.parameters, FullModel.coordinates)
    scheme = Scheme(
        model=FullModel.model, parameters=FullModel.parameters, data={"dataset1": dataset}
    )
    optimization_group = OptimizationGroup(scheme, FullModel.model.get_dataset_groups()["default"])

    result = optimization_group.create_result_data()["dataset1"]
    assert "global_matrix" in result
    assert "global_clp_label" in result

    clp = result.clp

    assert clp.shape == (4, 4)
    print(np.diagonal(clp))
    assert all(np.isclose(1.0, c) for c in np.diagonal(clp))
