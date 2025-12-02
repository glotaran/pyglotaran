from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glotaran.builtin.elements.kinetic.element import KineticElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.io import load_parameters
from glotaran.io import load_result
from glotaran.io import load_scheme
from glotaran.io import save_result
from glotaran.io import save_scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import RESULT

TEST_SCHEME_YML = """
# Just a comment
library:
    parallel:
        type: kinetic
        rates:
            (s1, s1): rates.1

experiments:
    myexp:
        datasets:
            kinetic_parallel:
                elements: [parallel]
                activations:
                    irf:
                        type: instant
                        compartments:
                            "s1": 1
"""


def test_parameter_group_copy():
    params = """
    a:
        - ["foo", 1, {non-negative: true, min: -1, max: 1, vary: false}]
        - 4
        - 5
    b:
        - 7
        - 8
    """
    parameters = load_parameters(params, format_name="yml_str")

    assert parameters.get("a.foo").value == 1
    assert parameters.get("a.foo").non_negative
    assert parameters.get("a.foo").minimum == -1
    assert parameters.get("a.foo").maximum == 1
    assert not parameters.get("a.foo").vary

    assert parameters.get("a.2").value == 4
    assert not parameters.get("a.2").non_negative
    assert parameters.get("a.2").minimum == -np.inf
    assert parameters.get("a.2").maximum == np.inf
    assert parameters.get("a.2").vary

    assert parameters.get("a.3").value == 5

    assert parameters.get("b.1").value == 7
    assert not parameters.get("b.1").non_negative
    assert parameters.get("b.1").minimum == -np.inf
    assert parameters.get("b.1").maximum == np.inf
    assert parameters.get("b.1").vary

    assert parameters.get("b.2").value == 8


def test_load_scheme():
    """Load scheme from string."""
    scheme = load_scheme(TEST_SCHEME_YML, format_name="yml_str")
    assert isinstance(scheme.library["parallel"], KineticElement)
    assert isinstance(
        scheme.experiments["myexp"].datasets["kinetic_parallel"], ActivationDataModel
    )


def test_save_scheme_from_string(tmp_path: Path):
    """Save and load scheme from file."""
    input_scheme = load_scheme(TEST_SCHEME_YML, format_name="yml_str")
    save_path = tmp_path / "test_scheme.yml"
    save_scheme(input_scheme, save_path)
    loaded_scheme = load_scheme(save_path)
    assert loaded_scheme.model_dump() == input_scheme.model_dump()


def test_save_scheme_from_file(tmp_path: Path):
    """YAML file roundtrip preserves comments."""
    input_path = tmp_path / "input_scheme.yml"
    input_path.write_text(TEST_SCHEME_YML)

    save_path = tmp_path / "test_scheme.yml"
    scheme = load_scheme(input_path)
    save_scheme(scheme, save_path)

    assert save_path.read_text() == TEST_SCHEME_YML

    # copyfile does fail if source and target are the same
    reloaded_scheme = load_scheme(save_path)
    assert reloaded_scheme.source_path == save_path
    save_scheme(reloaded_scheme, save_path, allow_overwrite=True)

    assert save_path.read_text() == TEST_SCHEME_YML


def test_save_scheme_from_file_edited(tmp_path: Path):
    """YAML file writes changes when scheme was changed in memory."""
    input_path = tmp_path / "input_scheme.yml"
    input_path.write_text(TEST_SCHEME_YML)

    save_path = tmp_path / "test_scheme.yml"
    scheme = load_scheme(input_path)
    scheme.experiments.pop("myexp")
    save_scheme(scheme, save_path)

    assert save_path.read_text() != TEST_SCHEME_YML
    assert load_scheme(save_path).experiments == {}


@pytest.mark.parametrize("result_file_name", ["result.yml", ""])
def test_result_round_tripping(tmp_path: Path, result_file_name: str):
    """Saving and loading Result via YAML preserves data."""
    save_path = tmp_path / result_file_name
    result_file_paths = save_result(RESULT, save_path)
    assert result_file_paths == [
        (tmp_path / "result.yml").as_posix(),
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
    ]
    assert all(Path(path).exists() for path in result_file_paths)
    loaded_result = load_result(save_path)
    for dataset in loaded_result.optimization_results.values():
        assert dataset.meta.weighted_root_mean_square_error is None
        assert dataset.meta.scale == 1

    loaded_result.scheme.optimize(loaded_result.optimized_parameters, loaded_result.input_data)
