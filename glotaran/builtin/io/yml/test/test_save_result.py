from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.io import save_result
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:
    from py.path import local as TmpDir

    from glotaran.project.result import Result


def test_save_result_yml(
    tmpdir: TmpDir,
    dummy_result: Result,  # noqa: F811
):
    """Check all files exist."""

    result_path = Path(tmpdir / "testresult.yml")
    save_result(file_name=result_path, format_name="yml", result=dummy_result)

    assert result_path.exists()
