"""Test deprecations for ``glotaran.project.project``."""

from __future__ import annotations

from pathlib import Path

import pytest

from glotaran.deprecation import GlotaranApiDeprecationWarning
from glotaran.optimization.optimize import optimize
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME


def test_project_generate_model():
    """Trow deprecation warning on accessing ``Result.number_of_data_points``."""
    print(SCHEME.data["dataset_1"])
    result = optimize(SCHEME, raise_exception=True)
    with pytest.warns(GlotaranApiDeprecationWarning) as records:
        result.number_of_data_points

        assert len(records) == 1
        assert Path(records[0].filename) == Path(
            __file__
        ), f"{Path(records[0].filename)=}, {Path(__file__)=}"
