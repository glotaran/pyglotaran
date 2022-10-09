import numpy as np
import pytest
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.test.models import SimpleTestModel
from glotaran.parameter import Parameters
from glotaran.project import Scheme


@pytest.fixture()
def dataset_one() -> xr.Dataset:
    global_axis = [1, 5, 6]
    model_axis = [5, 7, 9, 12]

    data = xr.DataArray(
        np.ones((4, 3)), coords=[("model", model_axis), ("global", global_axis)]
    ).to_dataset(name="data")

    data["weight"] = xr.ones_like(data.data) * 0.5
    return data


@pytest.fixture()
def dataset_two() -> xr.Dataset:
    global_axis = [0, 3, 7, 10]
    model_axis = [4, 11, 15]

    return xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", global_axis), ("model", model_axis)]
    ).to_dataset(name="data")


@pytest.fixture()
def scheme(dataset_one: xr.Dataset, dataset_two: xr.Dataset) -> Scheme:
    model = SimpleTestModel(
        **{
            "megacomplex": {"m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                },
                "dataset2": {
                    "megacomplex": ["m1"],
                },
            },
        }
    )
    print(model.validate())
    assert model.valid()

    parameters = Parameters.from_list([])

    data = {"dataset1": dataset_one, "dataset2": dataset_two}
    return Scheme(model, parameters, data, clp_link_tolerance=1)


@pytest.fixture()
def dataset_group(scheme: Scheme) -> DatasetGroup:
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    return dataset_group


def test_data_provider(
    dataset_one: xr.Dataset, dataset_two: xr.Dataset, scheme: Scheme, dataset_group: DatasetGroup
):
    data_provider = DataProvider(scheme, dataset_group)

    print(dataset_one.data)
    print(data_provider.get_data("dataset1"))
    assert data_provider.get_model_dimension("dataset1") == "model"
    assert data_provider.get_global_dimension("dataset1") == "global"
    assert np.array_equal(
        dataset_one.data * dataset_one.weight, data_provider.get_data("dataset1")
    )
    assert np.array_equal(dataset_one.weight, data_provider.get_weight("dataset1"))
    assert np.array_equal(dataset_one.coords["model"], data_provider.get_model_axis("dataset1"))
    assert np.array_equal(dataset_one.coords["global"], data_provider.get_global_axis("dataset1"))

    assert data_provider.get_model_dimension("dataset2") == "model"
    assert data_provider.get_global_dimension("dataset2") == "global"
    assert np.array_equal(dataset_two.data.T, data_provider.get_data("dataset2"))
    assert data_provider.get_weight("dataset2") is None
    assert np.array_equal(dataset_two.coords["model"], data_provider.get_model_axis("dataset2"))
    assert np.array_equal(dataset_two.coords["global"], data_provider.get_global_axis("dataset2"))


def test_data_provider_linked(
    dataset_one: xr.Dataset, dataset_two: xr.Dataset, scheme: Scheme, dataset_group: DatasetGroup
):
    data_provider = DataProviderLinked(scheme, dataset_group)

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
def test_data_provider_linking_methods(method: str, scheme: Scheme, dataset_group: DatasetGroup):
    scheme.clp_link_method = method  # type:ignore[assignment]
    data_provider = DataProviderLinked(scheme, dataset_group)

    #  global_axis1 = [1, 5, 6]
    #  global_axis2 = [0, 3, 7, 10]

    wanted_global_axis = [1, 3, 5, 6, 10]
    if method == "backward":
        wanted_global_axis = [0, 1, 3, 5, 6, 10]
    elif method == "forward":
        wanted_global_axis = [1, 3, 5, 6, 7, 10]
    assert np.array_equal(data_provider.aligned_global_axis, wanted_global_axis)
