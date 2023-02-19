from copy import deepcopy

import pytest

from glotaran.model import ClpRelation
from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.optimization.test.suites import TwoCompartmentDecay as suite
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
def test_relations(index_dependent, link_clp):
    model = deepcopy(suite.model)
    model.dataset_groups["default"].link_clp = link_clp
    model.megacomplex["m1"].is_index_dependent = index_dependent
    model.clp_relations.append(
        ClpRelation(**{"source": "s1", "target": "s2", "parameter": "3", "interval": (1, 1)})
    )
    parameters = Parameters.from_list([11e-4, 22e-5, 2])

    print("link_clp", link_clp, "index_dependent", index_dependent)  # T201
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        parameters,
        {"global": suite.global_axis, "model": suite.model_axis},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])
    optimization_group.calculate(parameters)

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
    assert clps.sel(clp_label="s2")[0] == clps.sel(clp_label="s1")[0] * 2
    assert clps.sel(clp_label="s2")[1] != clps.sel(clp_label="s1")[1] * 2
    assert "s2" in matrix.clp_labels

    # 1 compartment (s1 due to relation) * 2 items in global axis
    # + 1 compartment (s2 outside of interval) * 1 item in global axis
    assert optimization_group.number_of_clps == 3
