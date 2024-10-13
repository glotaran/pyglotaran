from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.objective import OptimizationObjective
from tests.optimization.data import TestDataModelConstantIndexDependent
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelGlobal


def test_single_data():
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    experiment = ExperimentModel(datasets={"test_data": data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = data_model.data["model"].size * data_model.data["global"].size
    assert penalty.size == data_size

    result = objective.get_result().data
    assert "test_data" in result
    result_data = result["test_data"]
    print(result_data)
    assert "test_ele" in result_data.elements
    element_result = result_data.elements["test_ele"]

    assert "concentrations" in element_result
    assert element_result.concentrations.shape == (
        data_model.data.model.size,
        1,
    )
    assert "amplitudes" in element_result
    assert element_result.amplitudes.shape == (
        data_model.data["global"].size,
        1,
    )
    assert result_data.residuals is not None
    assert result_data.input_data is not None
    assert result_data.input_data.shape == data_model.data.data.shape
    assert result_data.residuals.shape == data_model.data.data.shape


@pytest.mark.parametrize("weight", {True, False})
def test_global_data(weight: bool):
    data_model = deepcopy(TestDataModelGlobal)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    experiment = ExperimentModel(datasets={"test": data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = data_model.data["model"].size * data_model.data["global"].size
    assert penalty.size == data_size

    result = objective.get_result().data
    assert "test" in result

    result_data = result["test"]
    print(result_data)
    assert "matrix" in result_data
    assert result_data.matrix.shape == (
        (data_model.data["global"].size, data_model.data.model.size, 1)
        if weight
        else (data_model.data.model.size, 1)
    )
    assert "global_matrix" in result_data
    assert result_data.global_matrix.shape == (
        (data_model.data.model.size, data_model.data["global"].size, 1)
        if weight
        else (data_model.data["global"].size, 1)
    )
    assert "clp" in result_data
    assert result_data.clp.shape == (1, 1)
    assert result_data.residuals is not None
    assert result_data.residual.shape == data_model.data.data.shape


def test_multiple_data():
    data_model_one = deepcopy(TestDataModelConstantIndexIndependent)
    data_model_two = deepcopy(TestDataModelConstantIndexDependent)
    experiment = ExperimentModel(
        datasets={
            "independent": data_model_one,
            "dependent": data_model_two,
        }
    )
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, LinkedOptimizationData)

    penalty = objective.calculate()
    data_size_one = data_model_one.data["model"].size * data_model_one.data["global"].size
    data_size_two = data_model_two.data["model"].size * data_model_two.data["global"].size
    assert penalty.size == data_size_one + data_size_two

    result = objective.get_result().data

    assert "independent" in result
    result_data = result["independent"]
    print(result_data)
    assert "test_associated_concentrations_test_ele" in result_data
    assert result_data.test_associated_concentrations_test_ele.shape == (
        data_model_one.data.model.size,
        1,
    )
    assert "test_associated_amplitudes_test_ele" in result_data
    assert result_data.test_associated_amplitudes_test_ele.shape == (
        data_model_one.data["global"].size,
        1,
    )
    assert result_data.residuals is not None
    assert result_data.residual.shape == data_model_one.data.data.shape

    assert "dependent" in result
    result_data = result["dependent"]
    assert "test_associated_concentrations_test_ele_index_dependent" in result_data
    assert result_data.test_associated_concentrations_test_ele_index_dependent.shape == (
        data_model_two.data["global"].size,
        data_model_two.data.model.size,
        1,
    )
    assert "test_associated_amplitudes_test_ele_index_dependent" in result_data
    assert result_data.test_associated_amplitudes_test_ele_index_dependent.shape == (
        data_model_two.data["global"].size,
        1,
    )
    assert result_data.residuals is not None
    # this datamodel has transposed input
    assert result_data.residual.shape == data_model_two.data.data.T.shape


@pytest.mark.parametrize("weight", {True, False})
def test_result_data(weight: bool):
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    experiment = ExperimentModel(datasets={"test": data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = data_model.data["model"].size * data_model.data["global"].size
    assert penalty.size == data_size

    result = objective.get_result().data
    assert "test" in result

    result_data = result["test"]
    assert "root_mean_square_error" in result_data.attrs
    assert "data_left_singular_vectors" in result_data
    assert "data_right_singular_vectors" in result_data
    assert "residual_left_singular_vectors" in result_data
    assert "residual_right_singular_vectors" in result_data
    assert "residual_singular_values" in result_data
    assert np.array_equal(data_model.data.coords["model"], result_data.coords["model"])
    assert np.array_equal(data_model.data.coords["global"], result_data.coords["global"])
    assert result_data.input_data.shape == data_model.data.data.shape
    assert np.allclose(result_data.input_data, data_model.data.data)
    if weight:
        assert "weight" in result_data
        assert "weighted_root_mean_square_error" in result_data.attrs
        assert "weighted_residual" in result_data


def test_penalty():
    data_model_one = deepcopy(TestDataModelConstantIndexIndependent)
    data_model_two = deepcopy(TestDataModelConstantIndexDependent)
    experiment = ExperimentModel(
        datasets={
            "independent": data_model_one,
            "dependent": data_model_two,
        },
        clp_link_tolerance=1,
        clp_penalties=[
            EqualAreaPenalty(type="equal_area", source="c1", target="c2", parameter=2, weight=4)
        ],
    )
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, LinkedOptimizationData)

    penalty = objective.calculate()
    data_size_one = data_model_one.data["model"].size * data_model_one.data["global"].size
    data_size_two = data_model_two.data["model"].size * data_model_two.data["global"].size
    assert penalty.size == data_size_one + data_size_two + 1
    assert penalty[-1] == 20  # TODO: investigate
