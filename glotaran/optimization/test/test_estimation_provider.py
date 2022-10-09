import numpy as np
import pytest
import xarray as xr

from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderUnlinked
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


def test_estimation_provider_unlinked(scheme: Scheme):
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    data_provider = DataProvider(scheme, dataset_group)
    matrix_provider = MatrixProviderUnlinked(dataset_group, data_provider)
    estimation_provider = EstimationProviderUnlinked(dataset_group, data_provider, matrix_provider)
    matrix_provider.calculate()
    estimation_provider.estimate()

    full_penalty = estimation_provider.get_full_penalty()
    clp, residual = estimation_provider.get_result()

    dataset1_global_size = scheme.data["dataset1"].coords["global"].size
    dataset1_model_size = scheme.data["dataset1"].coords["model"].size
    dataset2_global_size = scheme.data["dataset2"].coords["global"].size
    dataset2_model_size = scheme.data["dataset2"].coords["model"].size

    assert (
        full_penalty.size
        == dataset1_global_size * dataset1_model_size + dataset2_global_size * dataset2_model_size
    )

    assert "dataset1" in clp
    assert clp["dataset1"].shape == (dataset1_global_size, 2)  # type:ignore[attr-defined]

    assert "dataset1" in residual
    assert (
        residual["dataset1"].shape  # type:ignore[attr-defined]
        == scheme.data["dataset1"].data.shape
    )

    assert "dataset2" in clp
    assert clp["dataset2"].shape == (dataset2_global_size, 2)  # type:ignore[attr-defined]

    assert "dataset2" in residual
    assert (
        residual["dataset2"].shape  # type:ignore[attr-defined]
        == scheme.data["dataset2"].data.T.shape
    )


def test_estimation_provider_linked(scheme: Scheme):
    dataset_group = scheme.model.get_dataset_groups()["default"]
    dataset_group.set_parameters(scheme.parameters)
    data_provider = DataProviderLinked(scheme, dataset_group)
    matrix_provider = MatrixProviderLinked(dataset_group, data_provider)
    estimation_provider = EstimationProviderLinked(dataset_group, data_provider, matrix_provider)
    matrix_provider.calculate()
    estimation_provider.estimate()

    full_penalty = estimation_provider.get_full_penalty()
    clp, residual = estimation_provider.get_result()

    dataset1_global_size = scheme.data["dataset1"].coords["global"].size
    dataset1_model_size = scheme.data["dataset1"].coords["model"].size
    dataset2_global_size = scheme.data["dataset2"].coords["global"].size
    dataset2_model_size = scheme.data["dataset2"].coords["model"].size

    assert (
        full_penalty.size
        == dataset1_global_size * dataset1_model_size + dataset2_global_size * dataset2_model_size
    )

    assert "dataset1" in clp
    assert clp["dataset1"].shape == (dataset1_global_size, 2)

    assert "dataset1" in residual
    assert residual["dataset1"].shape == scheme.data["dataset1"].data.shape

    assert "dataset2" in clp
    assert clp["dataset2"].shape == (dataset2_global_size, 2)

    assert "dataset2" in residual
    assert residual["dataset2"].shape == scheme.data["dataset2"].data.T.shape
