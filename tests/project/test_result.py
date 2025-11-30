from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import xarray as xr

from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter.parameter_history import ParameterHistory
from glotaran.parameter.parameters import Parameters
from glotaran.project.library import ModelLibrary
from glotaran.project.result import Result
from glotaran.project.scheme import Scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import RESULT
from glotaran.utils.io import chdir_context

if TYPE_CHECKING:
    from glotaran.io.interface import SavingOptions


def test_result_serde_default(tmp_path: Path):
    """Test serialization and deserialization of a result with default saving options."""
    serialized = RESULT.model_dump(
        mode="json",
        context={"save_folder": tmp_path},
    )

    assert serialized["initial_parameters"] == "initial_parameters.csv"
    assert (tmp_path / "initial_parameters.csv").is_file()
    assert serialized["optimized_parameters"] == "optimized_parameters.csv"
    assert (tmp_path / "optimized_parameters.csv").is_file()
    assert serialized["saving_options"] == SAVING_OPTIONS_DEFAULT | {"data_filter": []}

    optimization_info = serialized["optimization_info"]
    assert optimization_info["optimization_history"] == "optimization_history.csv"
    assert (tmp_path / "optimization_history.csv").is_file()
    assert optimization_info["parameter_history"] == "parameter_history.csv"
    assert (tmp_path / "parameter_history.csv").is_file()
    assert optimization_info["free_parameter_labels"] == [
        "rates.species_1",
        "rates.species_2",
        "rates.species_3",
        "irf.center",
        "irf.width",
    ]

    assert serialized["scheme"] == "scheme.yml"
    assert (tmp_path / serialized["scheme"]).is_file()

    optimization_results = serialized["optimization_results"]

    assert len(optimization_results) == 1

    sequential_results = optimization_results["sequential-decay"]
    assert len(sequential_results["elements"]) == 1
    assert sequential_results["elements"]["sequential"] == "sequential.nc"
    assert (tmp_path / "optimization_results/sequential-decay/elements/sequential.nc").is_file()
    assert len(sequential_results["activations"]) == 1
    assert sequential_results["activations"]["irf"] == "irf.nc"
    assert (tmp_path / "optimization_results/sequential-decay/activations/irf.nc").is_file()
    assert sequential_results["input_data"] == "input_data.nc"
    assert (tmp_path / "optimization_results/sequential-decay/input_data.nc").is_file()
    assert sequential_results["residuals"] == "residuals.nc"
    assert (tmp_path / "optimization_results/sequential-decay/residuals.nc").is_file()
    assert sequential_results["fitted_data"] == "fitted_data.nc"
    assert (tmp_path / "optimization_results/sequential-decay/fitted_data.nc").is_file()

    deserialized = Result.model_validate(serialized, context={"save_folder": tmp_path})
    assert deserialized.saving_options == SAVING_OPTIONS_DEFAULT
    assert isinstance(deserialized.scheme, Scheme)
    assert isinstance(deserialized.scheme.experiments["sequential-decay"], ExperimentModel)
    assert isinstance(deserialized.scheme.library, ModelLibrary)
    assert isinstance(deserialized.initial_parameters, Parameters)
    assert isinstance(deserialized.optimized_parameters, Parameters)
    assert isinstance(deserialized.optimization_info, OptimizationInfo)
    assert isinstance(deserialized.optimization_info.parameter_history, ParameterHistory)
    assert isinstance(deserialized.optimization_info.optimization_history, OptimizationHistory)
    assert deserialized.optimization_info.covariance_matrix is None
    assert deserialized.optimization_info.jacobian is None

    assert len(deserialized.optimization_results) == 1
    deserialized_sequential_results = deserialized.optimization_results["sequential-decay"]
    assert isinstance(deserialized_sequential_results.elements["sequential"], xr.Dataset)
    assert isinstance(deserialized_sequential_results.activations["irf"], xr.Dataset)
    assert isinstance(deserialized_sequential_results.input_data, xr.Dataset)
    assert isinstance(deserialized_sequential_results.residuals, xr.Dataset)
    assert isinstance(deserialized_sequential_results.fitted_data, xr.Dataset)


# We expect warnings about missing data when using minimal saving options
@pytest.mark.filterwarnings(r"ignore:Residuals must be set to calculate fitted data\.:UserWarning")
def test_result_serde_minimal(tmp_path: Path):
    """Test serialization and deserialization of a result with minimal saving options."""
    serialized = RESULT.model_dump(
        mode="json",
        context={"save_folder": tmp_path, "saving_options": SAVING_OPTIONS_MINIMAL},
    )

    assert serialized["initial_parameters"] == "initial_parameters.csv"
    assert (tmp_path / "initial_parameters.csv").is_file()
    assert serialized["optimized_parameters"] == "optimized_parameters.csv"
    assert (tmp_path / "optimized_parameters.csv").is_file()
    assert serialized["saving_options"] == SAVING_OPTIONS_MINIMAL | {
        "data_filter": list(SAVING_OPTIONS_MINIMAL["data_filter"])
    }

    optimization_info = serialized["optimization_info"]
    assert optimization_info["optimization_history"] == "optimization_history.csv"
    assert (tmp_path / "optimization_history.csv").is_file()
    assert optimization_info["parameter_history"] == "parameter_history.csv"
    assert (tmp_path / "parameter_history.csv").is_file()
    assert optimization_info["free_parameter_labels"] == [
        "rates.species_1",
        "rates.species_2",
        "rates.species_3",
        "irf.center",
        "irf.width",
    ]

    sequential_results = serialized["optimization_results"]["sequential-decay"]
    assert len(sequential_results["elements"]) == 0
    assert (tmp_path / "optimization_results/sequential-decay/elements").exists() is False
    assert len(sequential_results["activations"]) == 0
    assert (tmp_path / "optimization_results/sequential-decay/activations").exists() is False
    assert (tmp_path / "optimization_results/sequential-decay/residuals.nc").is_file() is False
    assert (tmp_path / "optimization_results/sequential-decay/fitted_data.nc").is_file() is False
    # The input data are saved since are an in memory dataset that wasn't saved before
    assert (tmp_path / "optimization_results/sequential-decay/input_data.nc").is_file() is True

    deserialized = Result.model_validate(serialized, context={"save_folder": tmp_path})
    assert deserialized.saving_options == SAVING_OPTIONS_MINIMAL
    assert isinstance(deserialized.scheme, Scheme)
    assert isinstance(deserialized.scheme.experiments["sequential-decay"], ExperimentModel)
    assert isinstance(deserialized.scheme.library, ModelLibrary)
    assert isinstance(deserialized.initial_parameters, Parameters)
    assert isinstance(deserialized.optimized_parameters, Parameters)
    assert isinstance(deserialized.optimization_info, OptimizationInfo)
    assert isinstance(deserialized.optimization_info.parameter_history, ParameterHistory)
    assert isinstance(deserialized.optimization_info.optimization_history, OptimizationHistory)
    assert deserialized.optimization_info.covariance_matrix is None
    assert deserialized.optimization_info.jacobian is None

    assert len(deserialized.optimization_results) == 1
    deserialized_sequential_results = deserialized.optimization_results["sequential-decay"]
    assert deserialized_sequential_results.elements == {}
    assert deserialized_sequential_results.activations == {}
    assert isinstance(deserialized_sequential_results.input_data, xr.Dataset)
    assert deserialized_sequential_results.residuals is None
    assert deserialized_sequential_results.fitted_data is None


def test_result_saving_options_are_used(tmp_path: Path):
    """Test that saving options provided in the context are used during serialization."""
    custom_saving_options: SavingOptions = {"parameters_format": "tsv"}

    serialized = RESULT.model_dump(
        mode="json",
        context={"save_folder": tmp_path, "saving_options": custom_saving_options},
    )

    assert serialized["initial_parameters"] == "initial_parameters.tsv"

    deserialized = Result.model_validate(serialized, context={"save_folder": tmp_path})
    assert isinstance(deserialized.initial_parameters, Parameters)
    assert isinstance(deserialized.optimized_parameters, Parameters)


@pytest.mark.xfail(reason="Needs to be fixed.")
@pytest.mark.parametrize("path_is_absolute", [True, False])
def test_saving(tmp_path: Path, path_is_absolute: bool):
    """Check all files exist."""
    warnings.warn("Test needs to be fixed.", stacklevel=2)
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
