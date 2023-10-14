from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.model import ExperimentModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.test.data import TestDataModelConstantIndexDependent
from glotaran.optimization.test.data import TestDataModelConstantIndexIndependent
from glotaran.optimization.test.data import TestDataModelGlobal


def test_single_data():
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    experiment = ExperimentModel(datasets={"test": data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = data_model.data["model"].size * data_model.data["global"].size
    assert penalty.size == data_size

    result = objective.get_result()
    assert "test" in result

    result_data = result["test"]
    assert "matrix" in result_data
    assert result_data.matrix.shape == (data_model.data.model.size, 1)
    assert "clp" in result_data
    assert result_data.clp.shape == (data_model.data["global"].size, 1)
    assert "residual" in result_data
    assert result_data.residual.shape == data_model.data.data.shape


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

    result = objective.get_result()
    assert "test" in result

    result_data = result["test"]
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
    assert "residual" in result_data
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

    result = objective.get_result()

    assert "independent" in result
    result_data = result["independent"]
    assert "matrix" in result_data
    assert result_data.matrix.shape == (data_model_one.data.model.size, 1)
    assert "clp" in result_data
    assert result_data.clp.shape == (data_model_one.data["global"].size, 1)
    assert "residual" in result_data
    assert result_data.residual.shape == data_model_one.data.data.shape

    assert "dependent" in result
    result_data = result["dependent"]
    assert "matrix" in result_data
    assert result_data.matrix.shape == (
        data_model_two.data["global"].size,
        data_model_two.data.model.size,
        1,
    )
    assert "clp" in result_data
    assert result_data.clp.shape == (data_model_two.data["global"].size, 1)
    assert "residual" in result_data
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

    result = objective.get_result()
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
    assert data_model.data.data.shape == result_data.data.shape
    print(data_model.data.data[0, 0], result_data.data[0, 0])  # T201
    assert np.allclose(data_model.data.data, result_data.data)
    if weight:
        assert "weight" in result_data
        assert "weighted_root_mean_square_error" in result_data.attrs
        assert "weighted_matrix" in result_data
        assert "weighted_residual" in result_data
