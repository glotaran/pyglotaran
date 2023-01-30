from copy import deepcopy

from glotaran.model import ExperimentModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.test.models import TestDataModelConstantIndexDependent
from glotaran.optimization.test.models import TestDataModelConstantIndexIndependent


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
    # this datamodel has transposed input
    assert result_data.residual.shape == data_model.data.data.T.shape


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
    # this datamodel has transposed input
    assert result_data.residual.shape == data_model_one.data.data.T.shape

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
    assert result_data.residual.shape == data_model_two.data.data.shape
