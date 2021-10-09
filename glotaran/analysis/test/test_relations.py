from copy import deepcopy

import pytest

from glotaran.analysis.optimization_group import OptimizationGroup
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import TwoCompartmentDecay as suite
from glotaran.model import Relation
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
def test_relations(index_dependent, link_clp):
    model = deepcopy(suite.model)
    model.dataset_group_models["default"].link_clp = link_clp
    model.megacomplex["m1"].is_index_dependent = index_dependent
    model.clp_relations.append(
        Relation.from_dict({"source": "s1", "target": "s2", "parameter": "3"})
    )
    parameters = ParameterGroup.from_list([11e-4, 22e-5, 2])

    print("link_clp", link_clp, "index_dependent", index_dependent)
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        parameters,
        {"global": suite.global_axis, "model": suite.model_axis},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"dataset1": dataset})
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])

    if index_dependent:
        reduced_matrix = (
            optimization_group.reduced_matrices[0]
            if link_clp
            else optimization_group.reduced_matrices["dataset1"][0]
        )
    else:
        reduced_matrix = optimization_group.reduced_matrices["dataset1"]
    matrix = (
        optimization_group.matrices["dataset1"][0]
        if index_dependent
        else optimization_group.matrices["dataset1"]
    )

    result_data = optimization_group.create_result_data()
    print(result_data)
    clps = result_data["dataset1"].clp

    assert "s2" not in reduced_matrix.clp_labels
    assert "s2" in clps.coords["clp_label"]
    assert clps.sel(clp_label="s2") == clps.sel(clp_label="s1") * 2
    assert "s2" in matrix.clp_labels
