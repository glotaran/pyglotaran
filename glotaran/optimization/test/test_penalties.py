from copy import deepcopy

import numpy as np
import pytest

from glotaran.model import EqualAreaPenalty
from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.optimization.test.suites import TwoCompartmentDecay as suite
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
def test_penalties(index_dependent, link_clp):
    model = deepcopy(suite.model)
    model.dataset_groups["default"].link_clp = link_clp
    model.megacomplex["m1"].is_index_dependent = index_dependent
    model.clp_penalties.append(
        EqualAreaPenalty(
            **{
                "source": "s1",
                "source_intervals": [(1, 20)],
                "target": "s2",
                "target_intervals": [(20, 45)],
                "parameter": "3",
                "weight": 10,
            }
        )
    )
    parameters = Parameters.from_list([11e-4, 22e-5, 2])

    global_axis = np.arange(50)

    print(f"{link_clp=}\n{index_dependent=}")  # T201
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        parameters,
        {"global": global_axis, "model": suite.model_axis},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])
    optimization_group.calculate(parameters)
    additional_penalty = optimization_group.get_additional_penalties()
    full_penalty = optimization_group.get_full_penalty()

    assert isinstance(additional_penalty, list)
    assert len(additional_penalty) == 1
    assert additional_penalty[0] != 0
    assert isinstance(full_penalty, np.ndarray)
    assert full_penalty.size == (suite.model_axis.size * global_axis.size) + len(
        additional_penalty
    )

    # 2 compartments * 50 items in global axis
    assert optimization_group.number_of_clps == 100
