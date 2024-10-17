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
    data_size = data_model.data["model_dim"].size * data_model.data["global_dim"].size
    assert penalty.size == data_size

    result = objective.get_result().optimization_results
    assert "test_data" in result
    result_data = result["test_data"]
    print(result_data)
    assert "test_ele" in result_data.elements
    element_result = result_data.elements["test_ele"]

    assert "concentrations" in element_result
    assert element_result.concentrations.shape == (
        data_model.data["model_dim"].size,
        1,
    )
    assert "amplitudes" in element_result
    assert element_result.amplitudes.shape == (
        data_model.data["global_dim"].size,
        1,
    )
    assert result_data.residuals is not None
    assert result_data.input_data is not None
    assert result_data.input_data.shape == data_model.data.data.shape
    assert result_data.residuals.shape == data_model.data.data.shape


@pytest.mark.parametrize("weight", {True, False})
def test_global_data(weight: bool):
    dataset_label = "dataset1"
    data_model = deepcopy(TestDataModelGlobal)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    experiment = ExperimentModel(datasets={dataset_label: data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)
    model_coord = data_model.data.coords["model_dim"]
    global_coord = data_model.data.coords["global_dim"]

    penalty = objective.calculate()
    data_size = model_coord.size * global_coord.size
    assert penalty.size == data_size

    result = objective.get_result().optimization_results
    assert dataset_label in result

    optimization_result = result[dataset_label]
    # TODO: something to figure out
    # when the full matrix is calculated, from the "elements" and "global_elements", the name
    # given to the results is the same as the dataset name
    element_result = optimization_result.elements[dataset_label]
    print(element_result)
    assert "model_concentrations" in element_result
    assert element_result["model_concentrations"].shape == (
        (global_coord.size, model_coord.size, 1) if weight else (model_coord.size, 1)
    )
    assert "global_concentrations" in element_result
    assert element_result["global_concentrations"].shape == (
        (model_coord.size, global_coord.size, 1) if weight else (global_coord.size, 1)
    )
    assert "amplitudes" in element_result
    assert element_result["amplitudes"].shape == (1, 1)
    assert optimization_result.residuals is not None
    assert optimization_result.residuals.shape == data_model.data.data.shape


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

    model_coord_one = data_model_one.data.coords["model_dim"]
    model_coord_two = data_model_two.data.coords["model_dim"]
    global_coord_one = data_model_one.data.coords["global_dim"]
    global_coord_two = data_model_two.data.coords["global_dim"]

    penalty = objective.calculate()
    data_size_one = model_coord_one.size * global_coord_one.size
    data_size_two = model_coord_two.size * global_coord_two.size
    assert penalty.size == data_size_one + data_size_two

    result = objective.get_result().optimization_results

    assert "independent" in result
    optimization_result_independent = result["independent"]
    assert optimization_result_independent.residuals is not None
    assert optimization_result_independent.residuals.shape == data_model_one.data.data.shape

    element_result_independent = optimization_result_independent.elements["test_ele"]
    assert "concentrations" in element_result_independent
    assert element_result_independent["concentrations"].shape == (
        model_coord_one.size,
        1,
    )
    assert "amplitudes" in element_result_independent
    assert element_result_independent["amplitudes"].shape == (
        global_coord_one.size,
        1,
    )

    assert "dependent" in result
    optimization_result_dependent = result["dependent"]
    assert optimization_result_dependent.residuals is not None
    # this datamodel has transposed input
    assert optimization_result_dependent.residuals.shape == data_model_two.data.data.T.shape

    element_result_dependent = optimization_result_dependent.elements["test_ele_index_dependent"]
    assert "concentrations" in element_result_dependent
    assert element_result_dependent["concentrations"].shape == (
        global_coord_two.size,
        model_coord_two.size,
        1,
    )
    assert "amplitudes" in element_result_dependent
    assert element_result_dependent["amplitudes"].shape == (
        global_coord_two.size,
        1,
    )


@pytest.mark.parametrize("weight", {True, False})
def test_result_data(weight: bool):
    dataset_label = "dataset1"
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    experiment = ExperimentModel(datasets={dataset_label: data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = (
        data_model.data.coords["model_dim"].size * data_model.data.coords["global_dim"].size
    )
    assert penalty.size == data_size

    result = objective.get_result().optimization_results
    assert dataset_label in result
    optimization_result = result[dataset_label]
    element_results = optimization_result.elements["test_ele"]
    assert "concentrations" in element_results
    assert "amplitudes" in element_results

    assert np.array_equal(data_model.data.coords["model_dim"], element_results.coords["model_dim"])
    assert np.array_equal(
        data_model.data.coords["global_dim"], element_results.coords["global_dim"]
    )
    assert optimization_result.input_data.shape == data_model.data.data.shape
    assert np.allclose(optimization_result.input_data, data_model.data.data)
    if weight:
        # TODO: find the lost weights
        # assert "weight" in result_data
        # assert "weighted_residual" in result_data
        pass


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
    data_size_one = (
        data_model_one.data.coords["model_dim"].size
        * data_model_one.data.coords["global_dim"].size
    )
    data_size_two = (
        data_model_two.data.coords["model_dim"].size
        * data_model_two.data.coords["global_dim"].size
    )
    assert penalty.size == data_size_one + data_size_two + 1
    assert penalty[-1] == 20  # TODO: investigate


if __name__ == "__main__":
    pytest.main([__file__])
