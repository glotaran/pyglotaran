from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran import __version__
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import save_dataset
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter.parameter_history import ParameterHistory
from glotaran.parameter.parameters import Parameters
from glotaran.plugin_system.data_io_registration import get_data_io
from glotaran.project.library import ModelLibrary
from glotaran.project.result import Result
from glotaran.project.scheme import Scheme
from glotaran.testing.plugin_system import monkeypatch_plugin_registry_data_io
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import RESULT
from glotaran.utils.io import chdir_context

if TYPE_CHECKING:
    from glotaran.io.interface import SavingOptions


def test_result_input_data():
    """Getting input data from result is the same as accessing it directly."""
    assert isinstance(RESULT.input_data, dict)
    assert "sequential-decay" in RESULT.input_data
    assert np.allclose(RESULT.input_data["sequential-decay"].data, DATASET.data)


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
    assert optimization_info["glotaran_version"] == __version__

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
    # Fit decomposition saved for optimization result
    assert "fit_decomposition" in sequential_results
    assert sequential_results["fit_decomposition"]["clp"] == "clp.nc"
    assert (tmp_path / "optimization_results/sequential-decay/fit_decomposition/clp.nc").is_file()
    assert sequential_results["fit_decomposition"]["matrix"] == "matrix.nc"
    assert (
        tmp_path / "optimization_results/sequential-decay/fit_decomposition/matrix.nc"
    ).is_file()

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
    assert optimization_info["glotaran_version"] == __version__

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


def test_result_extract_paths_from_serialization(tmp_path: Path):
    """Test that saving options provided in the context are used during serialization."""

    serialized = RESULT.model_dump(
        mode="json",
        context={"save_folder": tmp_path},
    )
    result_file_path = tmp_path / "result.yml"
    result_file_path.touch()

    assert Result.extract_paths_from_serialization(result_file_path, serialized) == [
        result_file_path.as_posix(),
        (tmp_path / "scheme.yml").as_posix(),
        (tmp_path / "initial_parameters.csv").as_posix(),
        (tmp_path / "optimized_parameters.csv").as_posix(),
        (tmp_path / "parameter_history.csv").as_posix(),
        (tmp_path / "optimization_history.csv").as_posix(),
        (tmp_path / "optimization_results/sequential-decay/input_data.nc").as_posix(),
        (tmp_path / "optimization_results/sequential-decay/residuals.nc").as_posix(),
        (tmp_path / "optimization_results/sequential-decay/fitted_data.nc").as_posix(),
        (tmp_path / "optimization_results/sequential-decay/elements/sequential.nc").as_posix(),
        (tmp_path / "optimization_results/sequential-decay/activations/irf.nc").as_posix(),
        (tmp_path / "optimization_results/sequential-decay/fit_decomposition/clp.nc").as_posix(),
        (
            tmp_path / "optimization_results/sequential-decay/fit_decomposition/matrix.nc"
        ).as_posix(),
    ]


def test_result_extract_paths_from_serialization_relative(tmp_path: Path):
    """Test that saving options provided in the context are used during serialization."""

    serialized = RESULT.model_dump(
        mode="json",
        context={"save_folder": tmp_path},
    )
    result_file_path = tmp_path / "result.yml"
    result_file_path.touch()

    with chdir_context(tmp_path):
        assert Result.extract_paths_from_serialization(result_file_path, serialized) == [
            "result.yml",
            "scheme.yml",
            "initial_parameters.csv",
            "optimized_parameters.csv",
            "parameter_history.csv",
            "optimization_history.csv",
            "optimization_results/sequential-decay/input_data.nc",
            "optimization_results/sequential-decay/residuals.nc",
            "optimization_results/sequential-decay/fitted_data.nc",
            "optimization_results/sequential-decay/elements/sequential.nc",
            "optimization_results/sequential-decay/activations/irf.nc",
            "optimization_results/sequential-decay/fit_decomposition/clp.nc",
            "optimization_results/sequential-decay/fit_decomposition/matrix.nc",
        ]


def test_result_extract_paths_from_serialization_minimal_save(tmp_path: Path):
    """Check that minimal paths can be correctly extracted from minimal save."""
    # This will serialize to a tuple with the plugin name rather than a relative path
    save_path = tmp_path / "original_data/input_data.foo"
    nc_plugin = get_data_io("nc")
    mock_plugin_name = "foo.FooNc"

    with monkeypatch_plugin_registry_data_io({mock_plugin_name: nc_plugin}):
        input_data = RESULT.optimization_results["sequential-decay"].input_data
        save_dataset(input_data, save_path, format_name=mock_plugin_name)
        input_data.attrs["io_plugin_name"] = mock_plugin_name

        assert input_data.attrs["source_path"] == save_path.as_posix()

        serialized = RESULT.model_dump(
            mode="json",
            context={"save_folder": tmp_path, "saving_options": SAVING_OPTIONS_MINIMAL},
        )
        assert serialized["optimization_results"]["sequential-decay"]["input_data"] == [
            "../../original_data/input_data.foo",
            mock_plugin_name,
        ]

        result_file_path = tmp_path / "result.yml"
        result_file_path.touch()

        assert Result.extract_paths_from_serialization(result_file_path, serialized) == [
            result_file_path.as_posix(),
            (tmp_path / "scheme.yml").as_posix(),
            (tmp_path / "initial_parameters.csv").as_posix(),
            (tmp_path / "optimized_parameters.csv").as_posix(),
            (tmp_path / "parameter_history.csv").as_posix(),
            (tmp_path / "optimization_history.csv").as_posix(),
            (tmp_path / "original_data/input_data.foo").as_posix(),
        ]


def test_result_save(tmp_path: Path):
    """Minimal check that save_result is properly wrapped."""
    result_file_paths = RESULT.save(tmp_path)

    assert len(result_file_paths) == 13
    assert result_file_paths[0] == (tmp_path / "result.yml").as_posix()
    assert (tmp_path / "result.yml").is_file()
    assert all(Path(path).exists() for path in result_file_paths)

    result_file_paths = RESULT.save(tmp_path / "minimal", saving_options=SAVING_OPTIONS_MINIMAL)
    assert len(result_file_paths) == 7
    assert result_file_paths[0] == (tmp_path / "minimal/result.yml").as_posix()
    assert (tmp_path / "minimal/result.yml").is_file()
    assert all(Path(path).exists() for path in result_file_paths)


if __name__ == "__main__":
    pytest.main([__file__])
