import collections

import numpy as np
import pytest

from glotaran.analysis.problem import Problem
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.simulation import simulate

from .mock import MultichannelMulticomponentDecay as suite


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
def test_problem(index_dependent, grouped):
    model = suite.model

    model.grouped = lambda: grouped
    model.index_dependent = lambda: index_dependent

    dataset = simulate(
        suite.sim_model, "dataset1", suite.wanted, {"e": suite.e_axis, "c": suite.c_axis}
    )
    scheme = Scheme(model=model, parameter=suite.initial, data={"dataset1": dataset})
    problem = Problem(scheme)

    assert problem.grouped == grouped
    assert problem.index_dependent == index_dependent

    bag = problem.bag

    if grouped:
        assert isinstance(bag, collections.deque)
        assert len(bag) == suite.e_axis.size
        assert problem.groups == {"dataset1": ["dataset1"]}
    else:
        assert isinstance(bag, dict)
        assert "dataset1" in bag

    clp_labels = problem.clp_labels
    matrices = problem.matrices

    if grouped:
        if index_dependent:
            assert isinstance(clp_labels, list)
            assert isinstance(matrices, list)
            assert all(isinstance(m, list) for m in problem.reduced_clp_labels)
            assert all(isinstance(m, np.ndarray) for m in problem.reduced_matrices)
        else:
            assert isinstance(clp_labels, dict)
            assert isinstance(matrices, dict)
            assert "dataset1" in problem.reduced_clp_labels
            assert "dataset1" in problem.reduced_matrices
            assert isinstance(problem.reduced_clp_labels["dataset1"], list)
            assert isinstance(problem.reduced_matrices["dataset1"], np.ndarray)
    else:
        if index_dependent:
            assert isinstance(clp_labels, dict)
            assert isinstance(matrices, dict)
            assert isinstance(problem.reduced_clp_labels, dict)
            assert isinstance(problem.reduced_matrices, dict)
            assert "dataset1" in problem.reduced_clp_labels
            assert "dataset1" in problem.reduced_matrices
            assert isinstance(problem.reduced_clp_labels["dataset1"], list)
            assert isinstance(problem.reduced_matrices["dataset1"], list)
            assert all(isinstance(c, list) for c in problem.reduced_clp_labels["dataset1"])
            assert all(isinstance(m, np.ndarray) for m in problem.reduced_matrices["dataset1"])
        else:
            assert isinstance(clp_labels, dict)
            assert isinstance(matrices, dict)
            assert "dataset1" in problem.reduced_clp_labels
            assert "dataset1" in problem.reduced_matrices
            assert isinstance(problem.reduced_clp_labels["dataset1"], list)
            assert isinstance(problem.reduced_matrices["dataset1"], np.ndarray)
