import numpy as np
import pytest
import xarray as xr

from glotaran.model import DataModel
from glotaran.model import ExperimentModel
from glotaran.optimization.data_provider import DataProvider
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


def test_data_provider_single_dataset(data_model_one: DataModel):
    experiment = ExperimentModel(datasets={"dataset1": data_model_one})
    data_provider = DataProvider(experiment)
    assert not data_provider.multiple_data

    dataset = data_model_one.data
    print(dataset.data)
    print(data_provider.get_data("dataset1"))
    assert data_provider.model_dimension == "model"
    assert data_provider.global_dimension == "global"
    assert np.array_equal(dataset.data * dataset.weight, data_provider.get_data("dataset1"))
    assert np.array_equal(dataset.weight, data_provider.get_weight("dataset1"))
    assert np.array_equal(dataset.coords["model"], data_provider.get_model_axis("dataset1"))
    assert np.array_equal(dataset.coords["global"], data_provider.get_global_axis("dataset1"))


def test_data_provider_multiple_dataset(data_model_one: DataModel, data_model_two: DataModel):
    experiment = ExperimentModel(
        datasets={"dataset1": data_model_one, "dataset2": data_model_two}, clp_link_tolerance=1
    )
    data_provider = DataProvider(experiment)
    assert data_provider.multiple_data

    dataset_one = data_model_one.data
    dataset_two = data_model_two.data

    assert "dataset1" in data_provider.group_definitions
    assert data_provider.group_definitions["dataset1"] == ["dataset1"]
    assert "dataset2" in data_provider.group_definitions
    assert data_provider.group_definitions["dataset2"] == ["dataset2"]
    assert "dataset1dataset2" in data_provider.group_definitions
    assert data_provider.group_definitions["dataset1dataset2"] == ["dataset1", "dataset2"]

    #  global_axis1 = [1, 5, 6]
    #  global_axis2 = [0, 3, 7, 10]

    assert np.array_equal(data_provider.aligned_global_axis, [1, 3, 5, 6, 10])

    assert data_provider.get_aligned_group_label(0) == "dataset1dataset2"
    assert data_provider.get_aligned_group_label(1) == "dataset2"
    assert data_provider.get_aligned_group_label(2) == "dataset1"
    assert data_provider.get_aligned_group_label(3) == "dataset1dataset2"
    assert data_provider.get_aligned_group_label(4) == "dataset2"

    assert np.array_equal(data_provider.get_aligned_dataset_indices(0), [0, 0])
    assert np.array_equal(data_provider.get_aligned_dataset_indices(1), [1])
    assert np.array_equal(data_provider.get_aligned_dataset_indices(2), [1])
    assert np.array_equal(data_provider.get_aligned_dataset_indices(3), [2, 2])
    assert np.array_equal(data_provider.get_aligned_dataset_indices(4), [3])

    dataset1_size = dataset_one.coords["model"].size
    dataset2_size = dataset_two.coords["model"].size

    assert data_provider.get_aligned_data(0).size == dataset1_size + dataset2_size
    assert data_provider.get_aligned_data(1).size == dataset2_size
    assert data_provider.get_aligned_data(2).size == dataset1_size
    assert data_provider.get_aligned_data(3).size == dataset1_size + dataset2_size
    assert data_provider.get_aligned_data(4).size == dataset2_size

    assert (
        data_provider.get_aligned_weight(0).size  # type:ignore[union-attr]
        == dataset1_size + dataset2_size
    )
    assert data_provider.get_aligned_weight(1) is None
    assert data_provider.get_aligned_weight(2).size == dataset1_size  # type:ignore[union-attr]
    assert (
        data_provider.get_aligned_weight(3).size  # type:ignore[union-attr]
        == dataset1_size + dataset2_size
    )
    assert data_provider.get_aligned_weight(4) is None


@pytest.mark.parametrize("method", ["nearest", "backward", "forward"])
def test_data_provider_linking_methods(
    method: str, data_model_one: DataModel, data_model_two: DataModel
):
    experiment = ExperimentModel(
        datasets={"dataset1": data_model_one, "dataset2": data_model_two},
        clp_link_tolerance=1,
        clp_link_method=method,
    )
    data_provider = DataProvider(experiment)
    assert data_provider.multiple_data

    #  global_axis1 = [1, 5, 6]
    #  global_axis2 = [0, 3, 7, 10]

    wanted_global_axis = [1, 3, 5, 6, 10]
    if method == "backward":
        wanted_global_axis = [0, 1, 3, 5, 6, 10]
    elif method == "forward":
        wanted_global_axis = [1, 3, 5, 6, 7, 10]
    assert np.array_equal(data_provider.aligned_global_axis, wanted_global_axis)
