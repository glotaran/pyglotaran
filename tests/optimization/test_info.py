"""Tests for the serializer module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.testing import assert_frame_equal

from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
from glotaran.parameter.parameters import Parameters

if TYPE_CHECKING:
    from pathlib import Path


def test_optimization_info_serialization(tmp_path: Path):
    """Test serialization and deserialization of OptimizationInfo partially to file."""
    parameter_history = ParameterHistory()
    parameter_history.append(Parameters.from_dict({"test": [4, 5, 6]}))

    optimization_history = OptimizationHistory(
        [
            {
                "iteration": 0,
                "nfev": 1,
                "cost": 2.0,
                "cost_reduction": 1.0,
                "step_norm": 0.1,
                "optimality": 0.5,
            }
        ]
    )
    info = OptimizationInfo(
        number_of_function_evaluations=1,
        success=False,
        termination_reason="Test",
        free_parameter_labels=["test.1", "test.2", "test.3"],
        parameter_history=parameter_history,
        optimization_history=optimization_history,
    )

    serialized = info.model_dump(mode="json", context={"save_folder": tmp_path})

    assert serialized["optimization_history"] == "optimization_history.csv"
    assert (tmp_path / "optimization_history.csv").is_file() is True
    assert serialized["parameter_history"] == "parameter_history.csv"
    assert (tmp_path / "parameter_history.csv").is_file() is True

    loaded = OptimizationInfo.model_validate(serialized, context={"save_folder": tmp_path})

    assert isinstance(loaded.optimization_history, OptimizationHistory)
    assert_frame_equal(loaded.optimization_history.data, optimization_history.data)
    assert isinstance(loaded.parameter_history, ParameterHistory)
    assert_frame_equal(loaded.parameter_history.to_dataframe(), parameter_history.to_dataframe())
