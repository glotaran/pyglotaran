import numpy as np
import pytest
import xarray as xr

from glotaran.model import DataModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.test.models import TestMegacomplexConstant


@pytest.fixture()
def data_model_one() -> DataModel:
    global_axis = [1, 5, 6]
    model_axis = [5, 7, 9, 12]

    data = xr.DataArray(
        np.ones((4, 3)), coords=[("model", model_axis), ("global", global_axis)]
    ).to_dataset(name="data")

    data["weight"] = xr.ones_like(data.data) * 0.5
    return DataModel(
        data=data,
        megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test",
                dimension="model",
                compartments=["b"],
                value="p2",
                is_index_dependent=False,
            )
        ],
    )


@pytest.fixture()
def data_model_two() -> DataModel:
    global_axis = [0, 3, 7, 10]
    model_axis = [4, 11, 15]

    data = xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", global_axis), ("model", model_axis)]
    ).to_dataset(name="data")
    return DataModel(
        data=data,
        megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test",
                dimension="model",
                compartments=["b"],
                value="p2",
                is_index_dependent=False,
            )
        ],
    )


@pytest.fixture()
def data_model_global() -> DataModel:
    global_axis = [0, 3, 7, 10]
    model_axis = [4, 11, 15]

    data = xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", global_axis), ("model", model_axis)]
    ).to_dataset(name="data")
    data["weight"] = xr.ones_like(data.data) * 0.5
    return DataModel(
        data=data,
        megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test",
                dimension="model",
                compartments=["b"],
                value="p2",
                is_index_dependent=False,
            )
        ],
        global_megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test-global",
                dimension="global",
                compartments=["b"],
                value="p2",
                is_index_dependent=False,
            )
        ],
    )


def test_optimization_data(data_model_one: DataModel):
    data = OptimizationData(data_model_one)

    dataset = data_model_one.data
    print(dataset.data)
    assert data.model_dimension == "model"
    assert data.global_dimension == "global"
    assert np.array_equal(dataset.data * dataset.weight, data.data)
    assert np.array_equal(dataset.weight, data.weight)
    assert np.array_equal(dataset.coords["model"], data.model_axis)
    assert np.array_equal(dataset.coords["global"], data.global_axis)


def test_optimization_data_global_model(data_model_global: DataModel):
    data = OptimizationData(data_model_global)

    dataset = data_model_global.data
    print(dataset.data)
    assert data.model_dimension == "model"
    assert data.global_dimension == "global"
    assert np.array_equal(
        dataset.data.data.T.flatten() * dataset.weight.data.T.flatten(), data.data
    )
    assert np.array_equal(dataset.weight.data.T.flatten(), data.weight)
    assert np.array_equal(dataset.coords["model"], data.model_axis)
    assert np.array_equal(dataset.coords["global"], data.global_axis)


def test_linked_optimization_data(data_model_one: DataModel, data_model_two: DataModel):
    all_data = {
        "dataset1": OptimizationData(data_model_one),
        "dataset2": OptimizationData(data_model_two),
    }
    tolerance, method = 1, "nearest"
    data = LinkedOptimizationData(all_data, tolerance, method)

    dataset_one = data_model_one.data
    dataset_two = data_model_two.data

    assert "dataset1" in data.group_definitions
    assert data.group_definitions["dataset1"] == ["dataset1"]
    assert "dataset2" in data.group_definitions
    assert data.group_definitions["dataset2"] == ["dataset2"]
    assert "dataset1dataset2" in data.group_definitions
    assert data.group_definitions["dataset1dataset2"] == ["dataset1", "dataset2"]

    #  global_axis1 = [1, 5, 6]
    #  global_axis2 = [0, 3, 7, 10]

    assert np.array_equal(data.global_axis, [1, 3, 5, 6, 10])

    assert data.group_labels[0] == "dataset1dataset2"
    assert data.group_labels[1] == "dataset2"
    assert data.group_labels[2] == "dataset1"
    assert data.group_labels[3] == "dataset1dataset2"
    assert data.group_labels[4] == "dataset2"

    assert np.array_equal(data.data_indices[0], [0, 0])
    assert np.array_equal(data.data_indices[1], [1])
    assert np.array_equal(data.data_indices[2], [1])
    assert np.array_equal(data.data_indices[3], [2, 2])
    assert np.array_equal(data.data_indices[4], [3])

    dataset1_size = dataset_one.coords["model"].size
    dataset2_size = dataset_two.coords["model"].size

    assert data.data[0].size == dataset1_size + dataset2_size
    assert data.data[1].size == dataset2_size
    assert data.data[2].size == dataset1_size
    assert data.data[3].size == dataset1_size + dataset2_size
    assert data.data[4].size == dataset2_size

    assert (
        data.weights[0].size  # type:ignore[union-attr]
        == dataset1_size + dataset2_size
    )
    assert data.weights[1] is None
    assert data.weights[2].size == dataset1_size  # type:ignore[union-attr]
    assert (
        data.weights[3].size  # type:ignore[union-attr]
        == dataset1_size + dataset2_size
    )
    assert data.weights[4] is None


@pytest.mark.parametrize("method", ["nearest", "backward", "forward"])
def test_linking_methods(method: str, data_model_one: DataModel, data_model_two: DataModel):
    all_data = {
        "dataset1": OptimizationData(data_model_one),
        "dataset2": OptimizationData(data_model_two),
    }
    tolerance = 1
    data = LinkedOptimizationData(all_data, tolerance, method)

    #  global_axis1 = [1, 5, 6]
    #  global_axis2 = [0, 3, 7, 10]

    wanted_global_axis = [1, 3, 5, 6, 10]
    if method == "backward":
        wanted_global_axis = [0, 1, 3, 5, 6, 10]
    elif method == "forward":
        wanted_global_axis = [1, 3, 5, 6, 7, 10]
    assert np.array_equal(data.global_axis, wanted_global_axis)
