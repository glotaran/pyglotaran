from __future__ import annotations

import pytest
from IPython.core.formatters import format_display_data

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import ThreeDatasetDecay as suite
from glotaran.project import Scheme
from glotaran.project.result import IncompleteResultError
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
        maximum_number_function_evaluations=1,
    )

    yield optimize(scheme)


def test_result_ipython_rendering(dummy_result: Result):
    """Autorendering in ipython"""

    rendered_obj = format_display_data(dummy_result)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("| Optimization Result")

    rendered_markdown_return = format_display_data(dummy_result.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("| Optimization Result")


def test_result_incomplete_exception(dummy_result: Result):
    """Raise error if required fields are missing."""

    with pytest.raises(IncompleteResultError) as excinfo:
        Result(1, True, "foo", "gta", ["1"])

    for mandatory_field, file_post_fix in [
        ("scheme", ""),
        ("initial_parameters", ""),
        ("optimized_parameters", ""),
        ("parameter_history", ""),
        ("data", "s"),
    ]:
        assert (
            f"Set either '{mandatory_field}' or '{mandatory_field}_file{file_post_fix}'."
            in str(excinfo.value)
        )
