from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.io import save_result
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:

    from glotaran.project.result import Result


def test_save_result_yml(
    tmp_path: Path,
    dummy_result: Result,  # noqa: F811
):
    """Check all files exist."""

    result_dir = tmp_path / "testresult"
    save_result(result_path=result_dir / "result.yml", result=dummy_result)

    assert (result_dir / "result.md").exists()
    assert (result_dir / "scheme.yml").exists()
    assert (result_dir / "result.yml").exists()
    assert (result_dir / "initial_parameters.csv").exists()
    assert (result_dir / "optimized_parameters.csv").exists()
    assert (result_dir / "dataset1.nc").exists()
    assert (result_dir / "dataset2.nc").exists()
    assert (result_dir / "dataset3.nc").exists()
