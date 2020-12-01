import collections

import numpy as np
import pytest

from glotaran.analysis.problem import Problem
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.simulation import simulate

from .mock import MultichannelMulticomponentDecay as suite


@pytest.fixture(
    scope="module", params=[[True, True], [True, False], [False, True], [False, False]]
)
def problem(request) -> Problem:
    model = suite.model
    model.grouped = lambda: request.param[0]
    model.index_dependent = lambda: request.param[1]

    dataset = simulate(
        suite.sim_model, "dataset1", suite.wanted, {"e": suite.e_axis, "c": suite.c_axis}
    )
    scheme = Scheme(model=model, parameter=suite.initial, data={"dataset1": dataset})
    return Problem(scheme)


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
            assert isinstance(problem.clp_labels, list)
            assert isinstance(problem.matrices, list)
            assert all(isinstance(m, list) for m in problem.reduced_clp_labels)
            assert all(isinstance(m, np.ndarray) for m in problem.reduced_matrices)
            assert len(problem.reduced_clp_labels) == suite.e_axis.size
            assert len(problem.reduced_matrices) == suite.e_axis.size
        else:
            assert isinstance(problem.clp_labels, dict)
            assert isinstance(problem.matrices, dict)
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

        assert isinstance(problem.matrices, dict)
        assert isinstance(problem.reduced_clp_labels["dataset1"], list)
        assert isinstance(problem.clp_labels, dict)
        assert "dataset1" in problem.reduced_clp_labels
        assert "dataset1" in problem.reduced_matrices


def test_problem_residuals(problem: Problem):
    problem.calculate_residual()
    if problem.grouped:
        assert isinstance(problem.reduced_clps, list)
        assert all(isinstance(c, np.ndarray) for c in problem.reduced_clps)
        assert len(problem.reduced_clps) == suite.e_axis.size
        assert isinstance(problem.full_clps, list)
        assert all(isinstance(c, np.ndarray) for c in problem.full_clps)
        assert len(problem.full_clps) == suite.e_axis.size
        assert isinstance(problem.residuals, list)
        assert all(isinstance(r, np.ndarray) for r in problem.residuals)
        assert len(problem.residuals) == suite.e_axis.size
    else:
        assert isinstance(problem.reduced_clps, dict)
        assert "dataset1" in problem.reduced_clps
        assert all(isinstance(c, np.ndarray) for c in problem.reduced_clps["dataset1"])
        assert len(problem.reduced_clps["dataset1"]) == suite.e_axis.size
        assert isinstance(problem.full_clps, dict)
        assert "dataset1" in problem.full_clps
        assert all(isinstance(c, np.ndarray) for c in problem.full_clps["dataset1"])
        assert len(problem.full_clps["dataset1"]) == suite.e_axis.size
        assert isinstance(problem.residuals, dict)
        assert "dataset1" in problem.residuals
        assert all(isinstance(r, np.ndarray) for r in problem.residuals["dataset1"])
        assert len(problem.residuals["dataset1"]) == suite.e_axis.size
    assert isinstance(problem.full_penalty, np.ndarray)
    assert problem.full_penalty.size == suite.c_axis.size * suite.e_axis.size


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
