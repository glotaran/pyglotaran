from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
from IPython.core.formatters import format_display_data

from glotaran.analysis.optimize import optimize
from glotaran.examples.sequential_spectral_decay import SCHEME
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import SavingOptions
from glotaran.project.result import IncompleteResultError
from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


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


def test_get_scheme(dummy_result: Result):
    scheme = dummy_result.get_scheme()
    assert "residual" not in dummy_result.scheme.data["dataset_1"]
    assert "residual" not in scheme.data["dataset_1"]
    assert all(scheme.parameters.to_dataframe() != dummy_result.scheme.parameters.to_dataframe())
    assert all(
        scheme.parameters.to_dataframe() == dummy_result.optimized_parameters.to_dataframe()
    )


@pytest.mark.parametrize("saving_options", [SAVING_OPTIONS_MINIMAL, SAVING_OPTIONS_DEFAULT])
def test_save_result(tmp_path: Path, saving_options: SavingOptions, dummy_result: Result):
    result_path = tmp_path / "test_result"
    dummy_result.save(str(result_path), saving_options=saving_options)
    files_must_exist = [
        "glotaran_result.yml",
        "scheme.yml",
        "model.yml",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "parameter_history.csv",
        "dataset_1.nc",
    ]
    files_must_not_exist = []
    if saving_options.report:
        files_must_exist.append("result.md")
    else:
        files_must_not_exist.append("result.md")

    for file in files_must_exist:
        assert (result_path / file).exists()

    for file in files_must_not_exist:
        assert not (result_path / file).exists()

    dataset_path = result_path / "dataset_1.nc"
    assert dataset_path.exists()
    dataset = xr.open_dataset(dataset_path)
    print(dataset)
    if saving_options.data_filter is not None:
        assert len(saving_options.data_filter) == len(dataset)
        assert all(d in dataset for d in saving_options.data_filter)


def test_recreate(dummy_result):
    recreated_result = dummy_result.recreate()
    assert recreated_result.success


def test_verify(dummy_result):
    assert dummy_result.verify()
