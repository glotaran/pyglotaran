from __future__ import annotations

import pytest
import xarray as xr
from IPython.core.formatters import format_display_data

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import ThreeDatasetDecay as suite
from glotaran.plugin_system.project_io_registration import SavingOptions
from glotaran.project import Scheme
from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""

    wanted_parameters = suite.wanted_parameters
    data = {}
    for i in range(3):
        global_axis = getattr(suite, "global_axis" if i == 0 else f"global_axis{i+1}")
        model_axis = getattr(suite, "model_axis" if i == 0 else f"model_axis{i+1}")

        data[f"dataset{i+1}"] = simulate(
            suite.sim_model,
            f"dataset{i+1}",
            wanted_parameters,
            {"global": global_axis, "model": model_axis},
        )

    scheme = Scheme(
        model=suite.model,
        parameters=suite.initial_parameters,
        data=data,
        maximum_number_function_evaluations=9,
    )

    yield optimize(scheme)


def test_get_scheme(dummy_result: Result):
    scheme = dummy_result.get_scheme()
    assert "residual" not in dummy_result.scheme.data["dataset1"]
    assert "residual" not in scheme.data["dataset1"]
    assert all(scheme.parameters.to_dataframe() != dummy_result.scheme.parameters.to_dataframe())
    assert all(
        scheme.parameters.to_dataframe() == dummy_result.optimized_parameters.to_dataframe()
    )


def test_result_ipython_rendering(dummy_result: Result):
    """Autorendering in ipython"""

    rendered_obj = format_display_data(dummy_result)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("| Optimization Result")

    rendered_markdown_return = format_display_data(dummy_result.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("| Optimization Result")


@pytest.mark.parametrize("level", ["minimal", "full"])
@pytest.mark.parametrize("data_filter", [None, ["clp"]])
@pytest.mark.parametrize("report", [True, False])
def test_save_result(tmp_path, level, data_filter, report, dummy_result: Result):
    result_path = tmp_path / "test_result"
    dummy_result.scheme.saving = SavingOptions(level=level, data_filter=data_filter, report=report)
    dummy_result.save(result_path)
    files_must_exist = [
        "glotaran_result.yml",
        "model.yml",
        "optimized_parameters.csv",
        "initial_parameters.csv",
    ]
    files_must_not_exist = []
    if report:
        files_must_exist.append("result.md")
    else:
        files_must_not_exist.append("result.md")

    for file in files_must_exist:
        assert (result_path / file).exists()

    for file in files_must_not_exist:
        assert not (result_path / file).exists()

    #  for i in range(1, 4):
    #      dataset_path = result_path / f"dataset{i}.nc"
    #      assert dataset_path.exists()
    #      dataset = xr.open_dataset(dataset_path)
    #      if data_filter is not None:
    #          assert len(data_filter) == len(dataset)
    #          assert all(d in dataset for d in data_filter)
    #      elif level == "minimal":
    #          data_filter = default_data_filters[level]
    #          assert len(data_filter) == len(dataset)
    #          assert all(d in dataset for d in data_filter)


def test_recreate(dummy_result):
    recreated_result = dummy_result.recreate()
    assert recreated_result.success


def test_verify(dummy_result):
    assert dummy_result.verify()
