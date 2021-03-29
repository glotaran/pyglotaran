from __future__ import annotations

import pytest

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import ThreeDatasetDecay as suite
from glotaran.project import Scheme


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""

    model = suite.model

    model.is_grouped = False
    model.is_index_dependent = False

    wanted_parameters = suite.wanted_parameters
    data = {}
    for i in range(3):
        e_axis = getattr(suite, "e_axis" if i == 0 else f"e_axis{i+1}")
        c_axis = getattr(suite, "c_axis" if i == 0 else f"c_axis{i+1}")

        data[f"dataset{i+1}"] = simulate(
            suite.sim_model, f"dataset{i+1}", wanted_parameters, {"e": e_axis, "c": c_axis}
        )
    scheme = Scheme(
        model=suite.model,
        parameters=suite.initial_parameters,
        data=data,
        maximum_number_function_evaluations=1,
    )

    yield optimize(scheme)
