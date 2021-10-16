from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.analysis.optimize import optimize
from glotaran.examples.sequential_spectral_decay import SCHEME
from glotaran.io import save_result

if TYPE_CHECKING:

    from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


def test_save_result_yml(
    tmp_path: Path,
    dummy_result: Result,
):
    """Check all files exist."""

    save_result(result_path=tmp_path / "result.yml", result=dummy_result)

    assert (tmp_path / "result.yml").exists()
