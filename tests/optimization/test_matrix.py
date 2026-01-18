from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.model.clp_constraint import ZeroConstraint
from glotaran.model.clp_relation import ClpRelation
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.parameter import Parameter
from tests.optimization.data import TestDataModelConstantIndexDependent
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelConstantThreeCompartments
from tests.optimization.data import TestDataModelGlobal

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel


@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize(
    "data_model", [TestDataModelConstantIndexIndependent, TestDataModelConstantIndexDependent]
)
def test_from_data(weight: bool, data_model: DataModel):
    data_model = deepcopy(data_model)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)
    assert matrix.array.shape == (
        (data.data.shape[1], data.data.shape[0], 1)
        if weight or data_model.elements[0].is_index_dependent
        else (data.data.shape[0], 1)
    )
    assert matrix.clp_axis == (["c2"] if data_model.elements[0].is_index_dependent else ["c1"])
    matrix_value = 2 if data_model.elements[0].is_index_dependent else 5
    if weight:
        matrix_value *= 0.5
    assert (matrix.array == matrix_value).all()


@pytest.mark.parametrize("weight", [True, False])
def test_from_global_data(weight: bool):
    data_model = deepcopy(TestDataModelGlobal)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    data = OptimizationData(data_model)
    _matrix, global_matrix, full_matrix = OptimizationMatrix.from_global_data(data)
    assert global_matrix.array.shape == (data.global_axis.size, 1)
    assert full_matrix.array.shape == (data.global_axis.size * data.model_axis.size, 1)
    assert full_matrix.clp_axis == ["c4@c4"]


def test_constraints():
    data_model = deepcopy(TestDataModelConstantThreeCompartments)
    data_model.elements[0].clp_constraints = [
        ZeroConstraint(type="zero", target="c3_3", interval=[(3, 7)])
    ]
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)

    assert matrix.array.shape == (5, 3)
    assert matrix.clp_axis == ["c3_1", "c3_2", "c3_3"]
    reduced_matrix = matrix.reduce(0, [])
    assert reduced_matrix.array.shape == (5, 3)
    assert reduced_matrix.clp_axis == ["c3_1", "c3_2", "c3_3"]
    reduced_matrix = matrix.reduce(3, [])
    assert reduced_matrix.array.shape == (5, 2)
    assert reduced_matrix.clp_axis == ["c3_1", "c3_2"]


def test_relations():
    data_model = deepcopy(TestDataModelConstantThreeCompartments)
    relations = [
        ClpRelation(
            source="c3_2", target="c3_3", parameter=Parameter(label="", value=3), interval=[(3, 7)]
        )
    ]
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)
    print(matrix)

    assert matrix.array.shape == (5, 3)
    assert matrix.clp_axis == ["c3_1", "c3_2", "c3_3"]

    reduced_matrix = matrix.at_index(0).reduce(0, relations)
    assert reduced_matrix.array.shape == (5, 3)
    assert reduced_matrix.clp_axis == ["c3_1", "c3_2", "c3_3"]
    reduced_matrix = matrix.at_index(3).reduce(3, relations)
    assert reduced_matrix.array.shape == (5, 2)
    assert reduced_matrix.clp_axis == ["c3_1", "c3_2"]
    assert np.array_equal(reduced_matrix.array[:, 1], matrix.array[:, 1] + matrix.array[:, 2] * 3)


def test_from_linked_data():
    data_model_one = deepcopy(TestDataModelConstantIndexIndependent)
    data_model_two = deepcopy(TestDataModelConstantIndexDependent)
    data_model_two.data["weight"] = xr.ones_like(data_model_two.data.data) * 0.5
    all_data = {
        "dataset1": OptimizationData(data_model_one),
        "dataset2": OptimizationData(data_model_two),
    }
    tolerance, method = 1, "nearest"
    linked_data = LinkedOptimizationData(all_data, tolerance, method, scales={"dataset2": 4})

    matrix_one = OptimizationMatrix.from_data(OptimizationData(data_model_one))
    matrix_two = OptimizationMatrix.from_data(OptimizationData(data_model_two))

    matrices = OptimizationMatrix.from_linked_data(linked_data)

    assert len(matrices) == linked_data.global_axis.size
    assert all(m.global_axis_size is None for m in matrices)
    assert all(not m.is_index_dependent for m in matrices)

    assert matrices[2].clp_axis == matrix_one.clp_axis
    assert np.array_equal(matrices[2].array, matrix_one.array)

    assert matrices[1].clp_axis == matrix_two.clp_axis
    assert np.array_equal(matrices[1].array, matrix_two.at_index(1).array * 4)

    assert matrices[0].clp_axis == matrix_one.clp_axis + matrix_two.clp_axis
    print(matrices[0].array[0 : matrix_one.model_axis_size, 0], matrix_one.array[:, 0])
    assert np.array_equal(
        matrices[0].array[0 : matrix_one.model_axis_size, 0], matrix_one.array[:, 0]
    )
    assert np.array_equal(
        matrices[0].array[
            matrix_one.model_axis_size : matrix_one.model_axis_size + matrix_two.model_axis_size, 1
        ],
        matrix_two.at_index(0).array[:, 0] * 4,
    )
