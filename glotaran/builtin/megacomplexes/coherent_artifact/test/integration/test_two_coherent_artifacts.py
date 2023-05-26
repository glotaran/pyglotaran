import shutil
from pathlib import Path

import numpy as np
import pytest

from glotaran.project import Project


def test_two_coherent_artifacts(tmp_path: Path):
    """Integration test based on simulated data with TIM."""
    shutil.copytree(
        Path(__file__).parent / "two_coherent_artifacts", tmp_path / "two_coherent_artifacts"
    )
    project = Project.open(tmp_path / "two_coherent_artifacts")
    result = project.optimize("two_coherent_artifacts", "two_coherent_artifacts")
    assert np.allclose(
        result.optimized_parameters.to_dataframe()[["value", "standard_error"]],
        project.load_parameters("expected").to_dataframe()[["value", "standard_error"]],
    )
    assert result.root_mean_square_error == pytest.approx(3.2101480e-06)
