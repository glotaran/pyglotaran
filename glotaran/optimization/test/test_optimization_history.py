"""Tests for ``glotaran.project.optimization_history``."""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from glotaran.optimization.optimization_history import OptimizationHistory


def test_optimization_history_init_no_data():
    """Empty DataFrame with correct columns and index."""
    result = OptimizationHistory()

    assert result.shape == (0, 5)
    assert all(result.columns == ["nfev", "cost", "cost_reduction", "step_norm", "optimality"])
    assert result.index.name == "iteration"


@pytest.mark.parametrize(
    "optimize_stdout, expected_df",
    (
        (
            "random string",
            pd.DataFrame(
                None,
                columns=["iteration", "nfev", "cost", "cost_reduction", "step_norm", "optimality"],
            ).set_index("iteration"),
        ),
        (
            dedent(
                """\
                   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
                       0              1         7.5834e+00                                    3.84e+01
                       1              2         7.5833e+00      1.37e-04       4.55e-05       1.26e-01
                       2              3         7.5833e+00      6.02e-11       6.44e-09       1.64e-05
                Both `ftol` and `xtol` termination conditions are satisfied.
                Function evaluations 3, initial cost 7.5834e+00, final cost 7.5833e+00, first-order optimality 1.64e-05.
                """  # noqa: E501
            ),
            pd.DataFrame(
                {
                    "iteration": [0, 1, 2],
                    "nfev": [1, 2, 3],
                    "cost": [7.5834, 7.5833, 7.5833],
                    "cost_reduction": [np.nan, 1.37e-4, 6.02e-11],
                    "step_norm": [np.nan, 4.55e-5, 6.44e-9],
                    "optimality": [38.4, 0.126, 1.64e-5],
                },
                columns=["iteration", "nfev", "cost", "cost_reduction", "step_norm", "optimality"],
            ).set_index("iteration"),
        ),
    ),
)
def test_optimization_history_init_from_stdout_str(
    optimize_stdout: str, expected_df: pd.DataFrame, tmp_path: Path
):
    """Parsing from ``optimize_stdout`` behaves gracefully and creates expected DataFrame."""
    result = OptimizationHistory.from_stdout_str(optimize_stdout)

    assert result.shape == expected_df.shape
    assert all(result.columns == ["nfev", "cost", "cost_reduction", "step_norm", "optimality"])
    assert result.index.name == "iteration"

    assert_frame_equal(result.data, expected_df)

    # pandas like access
    assert_series_equal(result.cost, expected_df.cost)
    assert_series_equal(result["cost"], expected_df["cost"])

    assert hasattr(result, "plot")

    # round tripping
    save_path = tmp_path / "optimization_history.csv"
    result.to_csv(save_path)

    round_tripped = OptimizationHistory.from_csv(save_path)

    assert_frame_equal(result._df, round_tripped._df)
    assert round_tripped.source_path == save_path.as_posix()
    assert result.source_path == round_tripped.source_path
