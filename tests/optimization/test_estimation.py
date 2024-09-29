from __future__ import annotations

from copy import deepcopy

import pytest

from glotaran.model.clp_constraint import ZeroConstraint
from glotaran.model.clp_relation import ClpRelation
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.parameter import Parameter
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelConstantThreeCompartments


@pytest.mark.parametrize(
    "residual_function", ("variable_projection", "non_negative_least_squares")
)
def test_calculate(residual_function: str):
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)
    matrices = [
        matrix.at_index(i).reduce(index, [], []) for i, index in enumerate(data.global_axis)
    ]
    estimations = [
        OptimizationEstimation.calculate(matrices[i].array, data.data[:, i], residual_function)
        for i in range(data.global_axis.size)
    ]
    assert all(e.clp.size == len(matrix.clp_axis) for e in estimations)
    assert all(e.residual.size == data.model_axis.size for e in estimations)


def test_resolve_clp():
    data_model = deepcopy(TestDataModelConstantThreeCompartments)
    data_model.elements[0].clp_constraints = [
        ZeroConstraint(type="zero", target="c3_1", interval=[(3, 7)])
    ]
    relations = [
        ClpRelation(
            source="c3_2", target="c3_3", parameter=Parameter(label="", value=3), interval=[(3, 7)]
        )
    ]
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)
    index = 3
    reduced_matrix = matrix.at_index(index).reduce(index, relations)
    estimation = OptimizationEstimation.calculate(
        reduced_matrix.array, data.data[:, 1], "variable_projection"
    )
    assert estimation.clp.size == 1

    resolved_estimation = estimation.resolve_clp(
        matrix.clp_axis, reduced_matrix.clp_axis, index, relations
    )
    assert resolved_estimation.clp.size == 3
