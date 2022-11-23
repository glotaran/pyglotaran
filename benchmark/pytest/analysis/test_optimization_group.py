from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex
from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.testing.plugin_system import monkeypatch_plugin_registry

if TYPE_CHECKING:
    from glotaran.model import DatasetModel

TEST_AXIS_MODEL_SIZE = 100
TEST_AXIS_MODEL = xr.DataArray(np.arange(0, TEST_AXIS_MODEL_SIZE))
TEST_AXIS_GLOBAL_SIZE = 100
TEST_AXIS_GLOBAL = xr.DataArray(np.arange(0, TEST_AXIS_GLOBAL_SIZE))
TEST_CLP_SIZE = 20
TEST_CLP_LABELS = [f"{i+1}" for i in range(TEST_CLP_SIZE)]
TEST_MATRIX_NON_INDEX_DEPENDENT = np.ones((TEST_AXIS_MODEL_SIZE, TEST_CLP_SIZE))
TEST_MATRIX_INDEX_DEPENDENT = np.ones((TEST_AXIS_GLOBAL_SIZE, TEST_AXIS_MODEL_SIZE, TEST_CLP_SIZE))
#  TEST_MATRIX = xr.DataArray(
#      np.ones((TEST_AXIS_MODEL_SIZE, TEST_CLP_SIZE)),
#      coords=(("test", TEST_AXIS_MODEL.data), ("clp_label", TEST_CLP_LABELS)),
#  )
TEST_DATA = xr.DataArray(
    np.ones((TEST_AXIS_GLOBAL_SIZE, TEST_AXIS_MODEL_SIZE)),
    coords=(("global", TEST_AXIS_GLOBAL.data), ("test", TEST_AXIS_MODEL.data)),
)
TEST_PARAMETERS = Parameters.from_list([])


@megacomplex()
class BenchmarkMegacomplex(Megacomplex):
    dimension: str = "test"
    type: str = "benchmark"
    is_index_dependent: bool

    def calculate_matrix(
        self,
        dataset_model,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ):
        if self.is_index_dependent is True:
            return TEST_CLP_LABELS, TEST_MATRIX_INDEX_DEPENDENT
        else:
            return TEST_CLP_LABELS, TEST_MATRIX_NON_INDEX_DEPENDENT

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


BenchmarkModel = Model.create_class_from_megacomplexes([BenchmarkMegacomplex])


@monkeypatch_plugin_registry(test_megacomplex={"benchmark": BenchmarkMegacomplex})
def setup_model(index_dependent, link_clp):
    model_dict = {
        "megacomplex": {"m1": {"type": "benchmark", "is_index_dependent": index_dependent}},
        "dataset_groups": {"default": {"link_clp": link_clp}},
        "dataset": {
            "dataset1": {"megacomplex": ["m1"]},
            "dataset2": {"megacomplex": ["m1"]},
            "dataset3": {"megacomplex": ["m1"]},
        },
    }
    return BenchmarkModel(**model_dict)


def setup_scheme(model):
    return Scheme(
        model=model,
        parameters=TEST_PARAMETERS,
        data={
            "dataset1": TEST_DATA,
            "dataset2": TEST_DATA,
            "dataset3": TEST_DATA,
        },
    )


def setup_optimization_group(scheme):
    return OptimizationGroup(scheme, scheme.model.get_dataset_groups()["default"])


def test_benchmark_align_data(benchmark):

    model = setup_model(False, True)
    assert model.valid()

    scheme = setup_scheme(model)

    benchmark(setup_optimization_group, scheme)


@pytest.mark.parametrize("link_clp", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_calculate_matrix(benchmark, link_clp, index_dependent):

    model = setup_model(index_dependent, link_clp)
    assert model.valid()

    scheme = setup_scheme(model)
    optimization_group = setup_optimization_group(scheme)

    benchmark(optimization_group._matrix_provider.calculate)


@pytest.mark.parametrize("link_clp", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_calculate_residual(benchmark, link_clp, index_dependent):

    model = setup_model(index_dependent, link_clp)
    assert model.valid()

    scheme = setup_scheme(model)
    optimization_group = setup_optimization_group(scheme)

    optimization_group._matrix_provider.calculate()

    benchmark(optimization_group._estimation_provider.estimate)


@pytest.mark.parametrize("link_clp", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_calculate_result_data(benchmark, link_clp, index_dependent):

    model = setup_model(index_dependent, link_clp)
    assert model.valid()

    scheme = setup_scheme(model)
    optimization_group = setup_optimization_group(scheme)

    optimization_group.calculate(scheme.parameters)

    benchmark(optimization_group.create_result_data)


#  @pytest.mark.skip(reason="To time consuming atm.")
@pytest.mark.parametrize("link_clp", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_optimize_20_runs(benchmark, link_clp, index_dependent):

    model = setup_model(index_dependent, link_clp)
    assert model.valid()

    scheme = setup_scheme(model)
    optimization_group = setup_optimization_group(scheme)

    @benchmark
    def run():

        for _ in range(20):
            optimization_group.calculate(scheme.parameters)

        optimization_group.create_result_data()
