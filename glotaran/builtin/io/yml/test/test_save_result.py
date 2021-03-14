import os

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import ThreeDatasetDecay as suite
from glotaran.io import write_result
from glotaran.project import Scheme


def test_optimization(tmpdir):
    model = suite.model

    model.is_grouped = False
    model.is_index_dependent = False

    wanted_parameters = suite.wanted_parameters
    data = {}
    for i in range(3):
        e_axis = getattr(suite, "e_axis" if i == 0 else f"e_axis{i+1}")
        c_axis = getattr(suite, "c_axis" if i == 0 else f"c_axis{i+1}")

        data[f"dataset{i+1}"] = simulate(
            suite.sim_model, f"dataset{i+1}", wanted_parameters, {"e": e_axis, "c": c_axis}
        )
    scheme = Scheme(
        model=suite.model,
        parameters=suite.initial_parameters,
        data=data,
        maximum_number_function_evaluations=1,
    )

    result = optimize(scheme)

    result_dir = os.path.join(tmpdir, "testresult")
    write_result(result_path=result_dir, format_name="yml", result=result)

    assert os.path.exists(os.path.join(result_dir, "result.md"))
    assert os.path.exists(os.path.join(result_dir, "scheme.yml"))
    assert os.path.exists(os.path.join(result_dir, "result.yml"))
    assert os.path.exists(os.path.join(result_dir, "initial_parameters.csv"))
    assert os.path.exists(os.path.join(result_dir, "optimized_parameters.csv"))
    assert os.path.exists(os.path.join(result_dir, "dataset1.nc"))
    assert os.path.exists(os.path.join(result_dir, "dataset2.nc"))
    assert os.path.exists(os.path.join(result_dir, "dataset3.nc"))
