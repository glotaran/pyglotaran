"""Test deprecations for ``glotaran.project.project``."""

from __future__ import annotations

from pathlib import Path

import pytest

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.project.project import Project


def test_project_generate_model(tmp_path: Path):
    """Trow deprecation warning on ``Project.generate_model`` call."""
    project = Project.open(tmp_path / "deprecated_project")
    records, _ = deprecation_warning_on_call_test_helper(
        project.generate_model,
        args=("generated_test_model", "decay_parallel", {"nr_compartments": 5}),
        raise_exception=True,
    )

    assert len(records) == 1


@pytest.mark.filterwarnings(
    "ignore::glotaran.deprecation.deprecation_utils.GlotaranApiDeprecationWarning"
)
def test_project_generate_parameters(tmp_path: Path):
    """Trow deprecation warning on ``Project.generate_parameters`` call."""
    project = Project.open(tmp_path / "deprecated_project")
    project.generate_model("generated_test_model", "decay_parallel", {"nr_compartments": 5})
    records, _ = deprecation_warning_on_call_test_helper(
        project.generate_parameters, args=("generated_test_model",), raise_exception=True
    )

    assert len(records) == 1
