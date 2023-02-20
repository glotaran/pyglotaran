from copy import deepcopy

import pytest

from glotaran.model import ZeroConstraint
from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.optimization.test.suites import TwoCompartmentDecay as suite
from glotaran.project import Scheme
from glotaran.simulation import simulate


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
def test_constraint(index_dependent, link_clp):
    model = deepcopy(suite.model)
    model.dataset_groups["default"].link_clp = link_clp
    model.megacomplex["m1"].is_index_dependent = index_dependent
    model.clp_constraints.append(ZeroConstraint(**{"target": "s2", "interval": (1, 1)}))

    print("link_clp", link_clp, "index_dependent", index_dependent)
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        suite.wanted_parameters,
        {"global": suite.global_axis, "model": suite.model_axis},
    )
    scheme = Scheme(model=model, parameters=suite.initial_parameters, data={"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])
    optimization_group.calculate(suite.initial_parameters)

    reduced_matrix = (
        optimization_group._matrix_provider.get_aligned_matrix_container(0)
        if link_clp
        else optimization_group._matrix_provider.get_prepared_matrix_container("dataset1", 0)
    )
    matrix = optimization_group._matrix_provider.get_matrix_container("dataset1")

    result_data = optimization_group.create_result_data()
    print(result_data)  # T201
    clps = result_data["dataset1"].clp

    assert "s2" not in reduced_matrix.clp_labels
    assert "s2" in clps.coords["clp_label"]
    assert clps.sel(clp_label="s2")[0] == 0
    assert clps.sel(clp_label="s2")[1] != 0
    assert "s2" in matrix.clp_labels

    # 1 compartment * 2 items in global axis
    # + 1 compartment * 1 item in global axis
    assert optimization_group.number_of_clps == 3
