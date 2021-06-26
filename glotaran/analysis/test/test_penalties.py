from copy import deepcopy

import numpy as np
import pytest

from glotaran.analysis.problem_grouped import GroupedProblem
from glotaran.analysis.problem_ungrouped import UngroupedProblem
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import TwoCompartmentDecay as suite
from glotaran.model import EqualAreaPenalty
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
def test_constraint(index_dependent, grouped):
    model = deepcopy(suite.model)
    model.megacomplex["m1"].is_index_dependent = index_dependent
    model.clp_area_penalties.append(
        EqualAreaPenalty.from_dict(
            {
                "source": "s1",
                "source_intervals": [(1, 20)],
                "target": "s2",
                "target_intervals": [(20, 45)],
                "parameter": "3",
                "weight": 10,
            }
        )
    )
    parameters = ParameterGroup.from_list([11e-4, 22e-5, 2])

    e_axis = np.arange(50)

    print("grouped", grouped, "index_dependent", index_dependent)
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        parameters,
        {"e": e_axis, "c": suite.c_axis},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"dataset1": dataset})
    problem = GroupedProblem(scheme) if grouped else UngroupedProblem(scheme)

    assert isinstance(problem.additional_penalty, np.ndarray)
    assert problem.additional_penalty.size == 1
    assert problem.additional_penalty[0] != 0
    assert isinstance(problem.full_penalty, np.ndarray)
    assert (
        problem.full_penalty.size
        == (suite.c_axis.size * e_axis.size) + problem.additional_penalty.size
    )
