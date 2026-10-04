from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_result
from glotaran.model.errors import GlotaranUserError
from glotaran.project import Scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from glotaran.testing.simulated_data.shared_decay import SIMULATION_PARAMETERS
from tests.project.conftest import LABEL

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from glotaran.parameter import Parameters
    from glotaran.project import Project
    from glotaran.project import Result

FULL_MODEL_SCHEME_DICT = {
    "library": SCHEME_DICT["library"],
    "experiments": {
        "full-model": {
            "datasets": {
                LABEL: {
                    "elements": ["sequential"],
                    "global_elements": ["spectral"],
                    "activations": SCHEME_DICT["experiments"][LABEL]["datasets"][LABEL][
                        "activations"
                    ],
                }
            }
        }
    },
}


def assert_results_equal(recomputed: Result, original: Result):
    """Fitted data, residuals and element arrays agree exactly."""
    for label, original_result in original.optimization_results.items():
        recomputed_result = recomputed.optimization_results[label]
        assert recomputed_result.fitted_data.equals(original_result.fitted_data)
        assert recomputed_result.residuals.equals(original_result.residuals)
        assert recomputed_result.elements.keys() == original_result.elements.keys()
        for element_label, element_result in original_result.elements.items():
            assert recomputed_result.elements[element_label].equals(element_result)


@pytest.mark.parametrize(
    ("scheme_dict", "parameters"),
    [(SCHEME_DICT, PARAMETERS), (FULL_MODEL_SCHEME_DICT, SIMULATION_PARAMETERS)],
    ids=["sequential", "kinetic-and-spectral"],
)
def test_recompute_reproduces_the_fit(
    project: Project, data: xr.Dataset, scheme_dict: dict, parameters: Parameters
):
    """Recompute from the record (scheme reloaded from scheme.yml) reproduces the fit exactly."""
    original = project.optimize(Scheme.from_dict(scheme_dict), parameters, {LABEL: data})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recomputed = project.recompute(original.record, {LABEL: data})

    assert_results_equal(recomputed, original)
    info, original_info = recomputed.optimization_info, original.optimization_info
    assert info.cost == original_info.cost
    assert info.degrees_of_freedom == original_info.degrees_of_freedom
    # The fit takes chi-square from SciPy's residual at its solution, but cost, parameters and
    # arrays from the last evaluated point (often a finite-difference Jacobian step).
    assert info.chi_square == pytest.approx(2 * original_info.cost, rel=1e-12)
    assert info.root_mean_square_error == pytest.approx(
        original_info.root_mean_square_error, rel=1e-9
    )
    assert recomputed.optimizer_settings == original.optimizer_settings
    assert recomputed.initial_parameters == parameters
    for parameter in original.optimized_parameters.all():
        assert recomputed.optimized_parameters.get(parameter.label).value == parameter.value

    recomputation = recomputed.recomputation
    assert recomputation["original_fit"]["id"] == original.record.id
    assert recomputation["original_fit"]["standard_errors"]["rates.species_1"] == (
        original.optimized_parameters.get("rates.species_1").standard_error
    )
    assert len(recomputation["original_fit"]["cost_history"]) > 0
    reconstruction = recomputation["reconstruction"]
    assert reconstruction["data_differences"] == []
    assert {value["relative_difference"] for value in reconstruction["drift"].values()} == {0.0}
    assert set(reconstruction["drift"]) == {
        "cost",
        f"{LABEL}.root_mean_square_error",
        f"{LABEL}.weighted_root_mean_square_error",
    }


def test_recompute_by_id_and_path_writes_no_record(
    project: Project, scheme: Scheme, data: xr.Dataset
):
    original = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)

    by_id = project.recompute(original.record.id, {LABEL: data})
    by_path = project.recompute(str(original.record.path), {LABEL: data})

    assert_results_equal(by_id, original)
    assert_results_equal(by_path, original)
    assert len(list(project.results_folder.iterdir())) == 1
    with pytest.raises(FileNotFoundError, match="No record found"):
        project.recompute("1999-01-01_00-00-00", {LABEL: data})


def test_recompute_with_different_data(project: Project, scheme: Scheme, data: xr.Dataset):
    """Changed data raise an error naming what changed, unless the mismatch is allowed."""
    original = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    changed = data.isel(time=slice(0, 200))

    with pytest.raises(GlotaranUserError, match=rf"{LABEL}: shape .*\n.*{LABEL}: time max"):
        project.recompute(original.record, {LABEL: changed})

    with pytest.warns(UserWarning, match="recomputed fit differs"):
        recomputed = project.recompute(original.record, {LABEL: changed}, allow_data_mismatch=True)
    differences = recomputed.recomputation["reconstruction"]["data_differences"]
    assert differences[0].startswith(f"{LABEL}: shape")


def test_recompute_warns_about_drift(project: Project, scheme: Scheme, data: xr.Dataset):
    """A recorded value that differs gives a warning; NaN is reported as not comparable."""
    original = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    record_file = original.record.path / "record.yml"
    content = load_dict(record_file, is_file=True)
    content["summary"]["cost"] *= 1.01
    content["summary"]["datasets"][LABEL]["root_mean_square_error"] = float("nan")
    write_dict(content, file_name=record_file)

    with pytest.warns(UserWarning, match="recomputed fit differs") as record:
        recomputed = project.recompute(original.record, {LABEL: data})

    message = str(record[0].message)
    assert "cost: " in message
    assert "(relative difference 9.9e-03)" in message
    assert f"{LABEL}.root_mean_square_error: nan" in message
    assert "(not comparable)" in message
    assert f"{LABEL}.weighted_root_mean_square_error" not in message
    drift = recomputed.recomputation["reconstruction"]["drift"]
    assert drift[f"{LABEL}.root_mean_square_error"]["relative_difference"] is None


def test_recomputed_result_saves_and_loads(
    project: Project, scheme: Scheme, data: xr.Dataset, tmp_path: Path
):
    original = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    recomputed = project.recompute(original.record, {LABEL: data})

    recomputed.save(tmp_path / "saved")
    loaded = load_result(tmp_path / "saved")

    assert loaded.recomputation["original_fit"]["id"] == original.record.id
    assert loaded.recomputation["reconstruction"]["drift"]["cost"]["relative_difference"] == 0
    assert np.isclose(
        loaded.recomputation["original_fit"]["cost_history"][-1],
        recomputed.recomputation["original_fit"]["cost_history"][-1],
        rtol=0,
    )
