from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from glotaran import __version__
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import load_result
from glotaran.io import save_dataset
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.info import OptimizerSettings
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
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
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
    assert deserialized_sequential_results.fitted_data.equals(
        RESULT.optimization_results["sequential-decay"].fitted_data
    )


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


def test_result_optimizer_settings_round_trip(tmp_path: Path):
    """Optimizer settings are stored on the result and in result.yml, defaults included.

    A tolerance of ``None`` (disabled in SciPy) is kept.
    """
    scheme = Scheme.from_dict(SCHEME_DICT)
    result = scheme.optimize(
        PARAMETERS,
        {"sequential-decay": DATASET},
        optimization_method="Dogbox",
        gtol=None,
        xtol=1e-6,
        maximum_number_function_evaluations=2,
        verbose=False,
    )
    expected = OptimizerSettings(
        optimization_method="Dogbox",
        ftol=1e-8,
        gtol=None,
        xtol=1e-6,
        maximum_number_function_evaluations=2,
    )
    assert result.optimizer_settings == expected

    result.save(tmp_path)
    assert load_result(tmp_path).optimizer_settings == expected


def test_result_weighted_input_data_round_trip(tmp_path: Path):
    """A dataset weight is kept in the input data and round-trips through save and load."""
    data = DATASET.copy()
    data["weight"] = xr.full_like(data.data, 0.5).transpose()
    result = Scheme.from_dict(SCHEME_DICT).optimize(
        PARAMETERS, {"sequential-decay": data}, verbose=False
    )
    input_data = result.input_data["sequential-decay"]
    assert isinstance(input_data, xr.Dataset)
    assert input_data.weight.dims == input_data.data.dims
    assert np.all(input_data.weight == 0.5)

    result.save(tmp_path)
    loaded = load_result(tmp_path).optimization_results["sequential-decay"]
    assert loaded.input_data.weight.equals(input_data.weight)
    assert loaded.fitted_data.equals(result.optimization_results["sequential-decay"].fitted_data)


def test_result_save_keeps_the_scheme_source_path(tmp_path: Path):
    """Saving a result does not repoint its scheme to the saved file."""
    result = Scheme.from_dict(SCHEME_DICT).optimize(
        PARAMETERS, {"sequential-decay": DATASET}, verbose=False
    )

    result.save(tmp_path / "first")
    (tmp_path / "first").rename(tmp_path / "moved")
    result.save(tmp_path / "second")

    assert result.scheme.source_path is None
    assert (tmp_path / "second" / "scheme.yml").is_file()


@pytest.mark.parametrize(
    "label",
    [
        "../outside",
        "a/b",
        "a\\b",
        "/outside",
        "C:\\outside",
        "C:outside",
        "..",
        ".",
        "",
        "a.",
        "a ",
        "a?",
        "a*",
        'a"',
        "a<b>",
        "a|b",
        "a\tb",
        "CON",
        "nul.txt",
        "com1",
        "LPT9 .nc",
    ],
)
def test_result_rejects_dataset_labels_that_are_no_file_names(label: str):
    """A dataset label that cannot name a folder in the result on every platform is rejected."""
    fields = {name: getattr(RESULT, name) for name in Result.model_fields}
    optimization_results = {label: RESULT.optimization_results["sequential-decay"]}

    with pytest.raises(ValidationError, match="Dataset label"):
        Result(**fields | {"optimization_results": optimization_results})


@pytest.mark.parametrize("label", ["sample 1", "CONSOLE", "nul_data", "a.b", "ΔA"])
def test_result_accepts_dataset_labels_that_are_file_names(label: str):
    """Labels that only resemble rejected ones are file names on every platform."""
    fields = {name: getattr(RESULT, name) for name in Result.model_fields}
    optimization_results = {label: RESULT.optimization_results["sequential-decay"]}

    result = Result(**fields | {"optimization_results": optimization_results})

    assert list(result.optimization_results) == [label]


def test_result_rejects_dataset_labels_that_differ_only_in_case():
    """Two dataset labels that would name the same folder on Windows or macOS are rejected."""
    fields = {name: getattr(RESULT, name) for name in Result.model_fields}
    optimization_result = RESULT.optimization_results["sequential-decay"]
    optimization_results = {"sample": optimization_result, "Sample": optimization_result}

    with pytest.raises(ValidationError, match="'sample' and 'Sample' differ only in case"):
        Result(**fields | {"optimization_results": optimization_results})


def test_load_result_rejects_a_dataset_label_before_reading_its_folder(tmp_path: Path):
    """Loading checks a dataset label before it reads the files of that dataset."""
    RESULT.save(tmp_path)
    result_file = tmp_path / "result.yml"
    result_file.write_text(
        result_file.read_text().replace("  sequential-decay:", "  ../outside:", 1)
    )

    # Reading from the missing folder would raise an error that does not name the label.
    with pytest.raises(ValidationError, match=re.escape("Dataset label '../outside'")):
        load_result(result_file)


def test_optimize_rejects_an_absolute_dataset_label(tmp_path: Path):
    """A fit with an absolute path as dataset label fails before the fit, without writing."""
    label = (tmp_path / "outside").as_posix()
    scheme_dict = deepcopy(SCHEME_DICT)
    datasets = scheme_dict["experiments"]["sequential-decay"]["datasets"]
    datasets[label] = datasets.pop("sequential-decay")

    with pytest.raises(GlotaranUserError, match="Dataset label"):
        Scheme.from_dict(scheme_dict).optimize(
            PARAMETERS, {label: DATASET}, maximum_number_function_evaluations=1, verbose=False
        )
    assert not (tmp_path / "outside").exists()


if __name__ == "__main__":
    pytest.main([__file__])
