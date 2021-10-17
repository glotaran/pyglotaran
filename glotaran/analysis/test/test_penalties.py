from copy import deepcopy

import numpy as np
import pytest

from glotaran.analysis.optimization_group import OptimizationGroup
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import TwoCompartmentDecay as suite
from glotaran.model import EqualAreaPenalty
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
def test_penalties(index_dependent, link_clp):
    model = deepcopy(suite.model)
    model.dataset_group_models["default"].link_clp = link_clp
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

    global_axis = np.arange(50)

    print(f"{link_clp=}\n{index_dependent=}")
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        parameters,
        {"global": global_axis, "model": suite.model_axis},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])

    assert isinstance(optimization_group.additional_penalty, np.ndarray)
    assert optimization_group.additional_penalty.size == 1
    assert optimization_group.additional_penalty[0] != 0
    assert isinstance(optimization_group.full_penalty, np.ndarray)
    assert (
        optimization_group.full_penalty.size
        == (suite.model_axis.size * global_axis.size) + optimization_group.additional_penalty.size
    )
