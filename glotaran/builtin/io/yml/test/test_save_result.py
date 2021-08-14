from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.analysis.optimize import optimize
from glotaran.examples.sequential import scheme
from glotaran.io import load_result
from glotaran.io import save_result

if TYPE_CHECKING:
    from py.path import local as TmpDir

    from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    scheme.maximum_number_function_evaluations = 1
    yield optimize(scheme)


def test_save_result_yml(
    tmpdir: TmpDir,
    dummy_result: Result,
):
    """Check all files exist."""

    result_path = Path(tmpdir / "testresult.yml")
    save_result(file_name=result_path, format_name="yml", result=dummy_result)

    assert result_path.exists()


def test_load_result(
    tmp_path,
    dummy_result: Result,
):
    path = tmp_path / "test_result"
    dummy_result.save(path)
    result_path = path / "glotaran_result.yml"
    print(result_path)  # noqa T001
    load_result(result_path)
