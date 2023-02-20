import numpy as np
import pytest
import xarray as xr

from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
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
            "megacomplex": {"m1": {"type": "simple-test-mc", "is_index_dependent": False}},
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
    parameters = Parameters.from_list([])

    data = {"dataset1": dataset_one, "dataset2": dataset_two}
    return Scheme(model, parameters, data, clp_link_tolerance=1)


def test_matrix_provider_unlinked_index_independent(scheme: Scheme):
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    data_provider = DataProvider(scheme, dataset_group)
    matrix_provider = MatrixProviderUnlinked(dataset_group, data_provider)
    matrix_provider.calculate()
    _, matrices = matrix_provider.get_result()

    assert "dataset1" in matrices
    assert matrices["dataset1"].shape == (scheme.data["dataset1"].model.size, 2)
    assert all(matrices["dataset1"].clp_label == ["s1", "s2"])

    assert "dataset2" in matrices
    assert matrices["dataset2"].shape == (scheme.data["dataset2"].model.size, 2)
    assert all(matrices["dataset2"].clp_label == ["s1", "s2"])

    # 2 compartments * 3 items in global axis of dataset1
    # + 2 compartments * 4 items in global axis of dataset2
    assert matrix_provider.number_of_clps == (2 * 3) + (2 * 4)


def test_matrix_provider_linked_index_independent(scheme: Scheme):
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    data_provider = DataProviderLinked(scheme, dataset_group)
    matrix_provider = MatrixProviderLinked(dataset_group, data_provider)
    matrix_provider.calculate()
    _, matrices = matrix_provider.get_result()

    dataset1_size = scheme.data["dataset1"].coords["model"].size
    dataset2_size = scheme.data["dataset2"].coords["model"].size

    assert "dataset1" in matrices
    assert matrices["dataset1"].shape == (dataset1_size, 2)
    assert all(matrices["dataset1"].clp_label == ["s1", "s2"])

    assert "dataset2" in matrices
    assert matrices["dataset2"].shape == (dataset2_size, 2)
    assert all(matrices["dataset2"].clp_label == ["s1", "s2"])

    assert matrix_provider.get_aligned_matrix_container(0).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(1).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(2).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(3).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(4).clp_labels == ["s1", "s2"]

    assert matrix_provider.get_aligned_matrix_container(0).matrix.shape == (
        dataset1_size + dataset2_size,
        2,
    )
    assert matrix_provider.get_aligned_matrix_container(1).matrix.shape == (dataset2_size, 2)
    assert matrix_provider.get_aligned_matrix_container(2).matrix.shape == (dataset1_size, 2)
    assert matrix_provider.get_aligned_matrix_container(3).matrix.shape == (
        dataset1_size + dataset2_size,
        2,
    )
    assert matrix_provider.get_aligned_matrix_container(4).matrix.shape == (dataset2_size, 2)

    # 2 compartments * 5 items in aligned global axis
    # See also: test_data_provider_linking_methods
    assert matrix_provider.number_of_clps == 2 * 5


def test_matrix_provider_unlinked_index_dependent(scheme: Scheme):
    scheme.model.megacomplex["m1"].is_index_dependent = True  # type:ignore[attr-defined]
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    data_provider = DataProvider(scheme, dataset_group)
    matrix_provider = MatrixProviderUnlinked(dataset_group, data_provider)
    matrix_provider.calculate()
    _, matrices = matrix_provider.get_result()

    dataset1_model_size = scheme.data["dataset1"].coords["model"].size
    dataset1_global_size = scheme.data["dataset1"].coords["global"].size
    dataset2_model_size = scheme.data["dataset2"].coords["model"].size
    dataset2_global_size = scheme.data["dataset2"].coords["global"].size

    assert "dataset1" in matrices
    assert matrices["dataset1"].shape == (dataset1_global_size, dataset1_model_size, 2)
    assert all(matrices["dataset1"].clp_label == ["s1", "s2"])

    assert "dataset2" in matrices
    assert matrices["dataset2"].shape == (dataset2_global_size, dataset2_model_size, 2)
    assert all(matrices["dataset2"].clp_label == ["s1", "s2"])

    # 2 compartments * 3 items in global axis of dataset1
    # + 2 compartments * 4 items in global axis of dataset2
    assert matrix_provider.number_of_clps == (2 * 3) + (2 * 4)


def test_matrix_provider_linked_index_dependent(scheme: Scheme):
    scheme.model.megacomplex["m1"].is_index_dependent = True  # type:ignore[attr-defined]
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    data_provider = DataProviderLinked(scheme, dataset_group)
    matrix_provider = MatrixProviderLinked(dataset_group, data_provider)
    matrix_provider.calculate()
    _, matrices = matrix_provider.get_result()

    dataset1_model_size = scheme.data["dataset1"].coords["model"].size
    dataset1_global_size = scheme.data["dataset1"].coords["global"].size
    dataset2_model_size = scheme.data["dataset2"].coords["model"].size
    dataset2_global_size = scheme.data["dataset2"].coords["global"].size

    assert "dataset1" in matrices
    assert matrices["dataset1"].shape == (dataset1_global_size, dataset1_model_size, 2)
    assert all(matrices["dataset1"].clp_label == ["s1", "s2"])

    assert "dataset2" in matrices
    assert matrices["dataset2"].shape == (dataset2_global_size, dataset2_model_size, 2)
    assert all(matrices["dataset2"].clp_label == ["s1", "s2"])

    assert matrix_provider.get_aligned_matrix_container(0).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(1).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(2).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(3).clp_labels == ["s1", "s2"]
    assert matrix_provider.get_aligned_matrix_container(4).clp_labels == ["s1", "s2"]

    assert matrix_provider.get_aligned_matrix_container(0).matrix.shape == (
        dataset1_model_size + dataset2_model_size,
        2,
    )
    assert matrix_provider.get_aligned_matrix_container(1).matrix.shape == (dataset2_model_size, 2)
    assert matrix_provider.get_aligned_matrix_container(2).matrix.shape == (dataset1_model_size, 2)
    assert matrix_provider.get_aligned_matrix_container(3).matrix.shape == (
        dataset1_model_size + dataset2_model_size,
        2,
    )
    assert matrix_provider.get_aligned_matrix_container(4).matrix.shape == (dataset2_model_size, 2)

    # 2 compartments * 5 items in aligned global axis
    # See also: test_data_provider_linking_methods
    assert matrix_provider.number_of_clps == 2 * 5
