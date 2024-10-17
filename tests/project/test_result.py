from __future__ import annotations

from pathlib import Path

import pytest

from glotaran.testing.simulated_data.sequential_spectral_decay import RESULT
from glotaran.utils.io import chdir_context


@pytest.mark.parametrize("path_is_absolute", (True, False))
def test_saving(tmp_path: Path, path_is_absolute: bool):
    """Check all files exist."""
    result_dir = tmp_path / "testresult" if path_is_absolute is True else Path("testresult")

    with chdir_context("." if path_is_absolute is True else tmp_path):
        RESULT.save(result_dir)

        assert (result_dir / "glotaran_result.yml").exists()
        assert (result_dir / "parameters_initial.csv").exists()
        assert (result_dir / "parameters_optimized.csv").exists()
        assert (result_dir / "optimization_history.csv").exists()
        assert (result_dir / "data" / "sequential-decay.nc").exists()


if __name__ == "__main__":
    # TODO: disable for now
    pytest.main([__file__])
