import collections

import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.problem import Problem
from glotaran.analysis.problem_grouped import GroupedProblem
from glotaran.analysis.problem_ungrouped import UngroupedProblem
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import MultichannelMulticomponentDecay as suite
from glotaran.analysis.test.models import SimpleTestModel
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


@pytest.fixture(
    scope="module", params=[[True, True], [True, False], [False, True], [False, False]]
)
def problem(request) -> Problem:
    model = suite.model
    model.is_grouped = request.param[0]
    model.is_index_dependent = request.param[1]

    dataset = simulate(
        suite.sim_model,
        "dataset1",
        suite.wanted_parameters,
        {"e": suite.e_axis, "c": suite.c_axis},
    )
    scheme = Scheme(model=model, parameters=suite.initial_parameters, data={"dataset1": dataset})
    return GroupedProblem(scheme) if request.param[0] else UngroupedProblem(scheme)


def test_problem(problem: Problem):
    assert problem.grouped == problem.model.is_grouped
    assert problem.index_dependent == problem.model.is_index_dependent


def test_problem_bag(problem: Problem):

    bag = problem.bag

    if problem.grouped:
        assert isinstance(bag, collections.deque)
        assert len(bag) == suite.e_axis.size
        assert problem.groups == {"dataset1": ["dataset1"]}
    else:
        assert isinstance(bag, dict)
        assert "dataset1" in bag


def test_problem_matrices(problem: Problem):
    problem.calculate_matrices()

    if problem.grouped:
        if problem.index_dependent:
            assert all(isinstance(m, list) for m in problem.reduced_clp_labels)
            assert all(isinstance(m, np.ndarray) for m in problem.reduced_matrices)
            assert len(problem.reduced_clp_labels) == suite.e_axis.size
            assert len(problem.reduced_matrices) == suite.e_axis.size
        else:
            assert "dataset1" in problem.reduced_clp_labels
            assert "dataset1" in problem.reduced_matrices
            assert isinstance(problem.reduced_clp_labels["dataset1"], list)
            assert isinstance(problem.reduced_matrices["dataset1"], np.ndarray)
    else:
        if problem.index_dependent:
            assert isinstance(problem.reduced_clp_labels, dict)
            assert isinstance(problem.reduced_matrices, dict)
            assert isinstance(problem.reduced_matrices["dataset1"], list)
            assert all(isinstance(c, list) for c in problem.reduced_clp_labels["dataset1"])
            assert all(isinstance(m, np.ndarray) for m in problem.reduced_matrices["dataset1"])
        else:
            assert isinstance(problem.reduced_matrices["dataset1"], np.ndarray)

        assert isinstance(problem.clp_labels, dict)
        assert isinstance(problem.matrices, dict)
        assert isinstance(problem.reduced_clp_labels["dataset1"], list)
        assert "dataset1" in problem.reduced_clp_labels
        assert "dataset1" in problem.reduced_matrices


def test_problem_residuals(problem: Problem):
    problem.calculate_residual()
    if problem.grouped:
        assert isinstance(problem.residuals, list)
        assert all(isinstance(r, np.ndarray) for r in problem.residuals)
        assert len(problem.residuals) == suite.e_axis.size
    else:
        assert isinstance(problem.residuals, dict)
        assert "dataset1" in problem.residuals
        assert all(isinstance(r, np.ndarray) for r in problem.residuals["dataset1"])
        assert len(problem.residuals["dataset1"]) == suite.e_axis.size
    assert isinstance(problem.reduced_clps, dict)
    assert "dataset1" in problem.reduced_clps
    assert all(isinstance(c, np.ndarray) for c in problem.reduced_clps["dataset1"])
    assert len(problem.reduced_clps["dataset1"]) == suite.e_axis.size
    assert isinstance(problem.clps, dict)
    assert "dataset1" in problem.clps
    assert all(isinstance(c, np.ndarray) for c in problem.clps["dataset1"])
    assert len(problem.clps["dataset1"]) == suite.e_axis.size
    assert isinstance(problem.additional_penalty, np.ndarray)
    assert problem.additional_penalty.size == 1
    assert problem.additional_penalty[0] == 0.1
    assert isinstance(problem.full_penalty, np.ndarray)
    assert (
        problem.full_penalty.size
        == (suite.c_axis.size * suite.e_axis.size) + problem.additional_penalty.size
    )


def test_problem_result_data(problem: Problem):

    data = problem.create_result_data()

    assert "dataset1" in data

    dataset = data["dataset1"]

    assert "clp_label" in dataset.coords
    assert np.array_equal(dataset.clp_label, ["s1", "s2", "s3", "s4"])

    assert problem.model.global_dimension in dataset.coords
    assert np.array_equal(dataset.coords[problem.model.global_dimension], suite.e_axis)

    assert problem.model.model_dimension in dataset.coords
    assert np.array_equal(dataset.coords[problem.model.model_dimension], suite.c_axis)

    assert "matrix" in dataset
    matrix = dataset.matrix
    if problem.index_dependent:
        assert len(matrix.shape) == 3
        assert matrix.shape[0] == suite.e_axis.size
        assert matrix.shape[1] == suite.c_axis.size
        assert matrix.shape[2] == 4
    else:
        assert len(matrix.shape) == 2
        assert matrix.shape[0] == suite.c_axis.size
        assert matrix.shape[1] == 4

    assert "clp" in dataset
    clp = dataset.clp
    assert len(clp.shape) == 2
    assert clp.shape[0] == suite.e_axis.size
    assert clp.shape[1] == 4

    assert "weighted_residual" in dataset
    assert dataset.data.shape == dataset.weighted_residual.shape

    assert "residual" in dataset
    assert dataset.data.shape == dataset.residual.shape

    assert "residual_singular_values" in dataset
    assert "weighted_residual_singular_values" in dataset


def test_prepare_data():
    model_dict = {
        "dataset": {
            "dataset1": {
                "megacomplex": [],
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
        coords={"e": global_axis, "c": model_axis},
        dims=("e", "c"),
    )

    scheme = Scheme(model, parameters, {"dataset1": dataset})
    problem = Problem(scheme)

    data = problem.data["dataset1"]
    print(data)
    assert "data" in data
    assert "weight" in data

    assert data.data.shape == (model_axis.size, global_axis.size)
    assert data.data.shape == data.weight.shape
    assert np.all(data.weight.sel(e=slice(0, 200), c=slice(4, 8)).values == 0.5)
    assert np.all(data.weight.sel(c=slice(0, 3)).values == 1)

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
    problem = Problem(scheme)
    data = problem.data["dataset1"]
    assert np.all(data.weight.sel(e=slice(0, 200), c=slice(4, 8)).values == 0.5 * 0.2)
    assert np.all(data.weight.sel(c=slice(0, 3)).values == 0.2)

    with pytest.warns(
        UserWarning,
        match="Ignoring model weight for dataset 'dataset1'"
        " because weight is already supplied by dataset.",
    ):
        Problem(Scheme(model, parameters, {"dataset1": data}))
