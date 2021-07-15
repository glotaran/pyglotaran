import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.problem_grouped import GroupedProblem
from glotaran.analysis.problem_ungrouped import UngroupedProblem
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme

TEST_AXIS_MODEL_SIZE = 100
TEST_AXIS_MODEL = xr.DataArray(np.arange(0, TEST_AXIS_MODEL_SIZE))
TEST_AXIS_GLOBAL_SIZE = 100
TEST_AXIS_GLOBAL = xr.DataArray(np.arange(0, TEST_AXIS_GLOBAL_SIZE))
TEST_CLP_SIZE = 20
TEST_CLP_LABELS = [f"{i+1}" for i in range(TEST_CLP_SIZE)]
TEST_MATRIX = np.ones((TEST_AXIS_MODEL_SIZE, TEST_CLP_SIZE))
#  TEST_MATRIX = xr.DataArray(
#      np.ones((TEST_AXIS_MODEL_SIZE, TEST_CLP_SIZE)),
#      coords=(("test", TEST_AXIS_MODEL.data), ("clp_label", TEST_CLP_LABELS)),
#  )
TEST_DATA = xr.DataArray(
    np.ones((TEST_AXIS_GLOBAL_SIZE, TEST_AXIS_MODEL_SIZE)),
    coords=(("global", TEST_AXIS_GLOBAL.data), ("test", TEST_AXIS_MODEL.data)),
)
TEST_PARAMETER = ParameterGroup.from_list([])


@megacomplex(dimension="test", properties={"is_index_dependent": bool})
class BenchmarkMegacomplex(Megacomplex):
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        return TEST_CLP_LABELS, TEST_MATRIX

    def index_dependent(self, dataset_model):
        return self.is_index_dependent

    def finalize_data(
        self,
        dataset_model,
        data,
        full_model: bool = False,
        as_global: bool = False,
    ):
        pass


def setup_model(index_dependent):
    model_dict = {
        "megacomplex": {"m1": {"is_index_dependent": index_dependent}},
        "dataset": {
            "dataset1": {"megacomplex": ["m1"]},
            "dataset2": {"megacomplex": ["m1"]},
            "dataset3": {"megacomplex": ["m1"]},
        },
    }
    return Model.from_dict(
        model_dict,
        megacomplex_types={"benchmark": BenchmarkMegacomplex},
        default_megacomplex_type="benchmark",
    )


def setup_scheme(model):
    return Scheme(
        model=model,
        parameters=TEST_PARAMETER,
        data={
            "dataset1": TEST_DATA,
            "dataset2": TEST_DATA,
            "dataset3": TEST_DATA,
        },
    )


def setup_problem(scheme, grouped):
    return GroupedProblem(scheme) if grouped else UngroupedProblem(scheme)


def test_benchmark_bag_creation(benchmark):

    model = setup_model(False)
    assert model.valid()

    scheme = setup_scheme(model)
    problem = setup_problem(scheme, True)

    benchmark(problem.init_bag)


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_calculate_matrix(benchmark, grouped, index_dependent):

    model = setup_model(index_dependent)
    assert model.valid()

    scheme = setup_scheme(model)
    problem = setup_problem(scheme, grouped)

    if grouped:
        problem.init_bag()

    benchmark(problem.calculate_matrices)


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_calculate_residual(benchmark, grouped, index_dependent):

    model = setup_model(index_dependent)
    assert model.valid()

    scheme = setup_scheme(model)
    problem = setup_problem(scheme, grouped)

    if grouped:
        problem.init_bag()
    problem.calculate_matrices()

    benchmark(problem.calculate_residual)


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_calculate_result_data(benchmark, grouped, index_dependent):

    model = setup_model(index_dependent)
    assert model.valid()

    scheme = setup_scheme(model)
    problem = setup_problem(scheme, grouped)

    if grouped:
        problem.init_bag()
    problem.calculate_matrices()
    problem.calculate_residual()

    benchmark(problem.create_result_data)


#  @pytest.mark.skip(reason="To time consuming atm.")
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_benchmark_optimize_20_runs(benchmark, grouped, index_dependent):

    model = setup_model(index_dependent)
    assert model.valid()

    scheme = setup_scheme(model)
    problem = setup_problem(scheme, grouped)

    @benchmark
    def run():
        if grouped:
            problem.init_bag()

        for _ in range(20):
            problem.reset()
            problem.full_penalty

        problem.create_result_data()
