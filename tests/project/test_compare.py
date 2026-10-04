from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from glotaran.project import Scheme
from glotaran.project import record as record_module
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from tests.project.conftest import LABEL

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from glotaran.project import Project


def fit_from(
    project: Project,
    scheme: Scheme,
    data: xr.Dataset,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs,  # noqa: ANN003
):
    source.touch()
    monkeypatch.setattr(record_module, "detect_source", lambda: source)
    return project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False, **kwargs)


def test_list_results(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """Records from two notebooks, with the data replaced partway, among v0.7 and damaged ones."""
    first = fit_from(project, scheme, data, project.folder / "a.ipynb", monkeypatch)
    second = fit_from(
        project, scheme, data, project.folder / "b.ipynb", monkeypatch, name="target"
    )
    third = fit_from(
        project, scheme, data.isel(time=slice(0, 100)), project.folder / "a.ipynb", monkeypatch
    )
    (project.results_folder / "v07_run").mkdir()
    (project.results_folder / "v07_run" / "result.yml").write_text("glotaran_version: 0.7.4\n")
    (project.results_folder / "damaged").mkdir()
    (project.results_folder / "damaged" / "record.yml").write_text("id: [unclosed\n")

    with pytest.warns(UserWarning, match="Skipped the damaged record"):
        table = project.list_results()

    assert list(table.index) == [first.record.id, second.record.id, third.record.id]
    assert list(table["source"]) == ["../../a.ipynb", "../../b.ipynb", "../../a.ipynb"]
    assert list(table["status"]) == ["success"] * 3
    assert list(table[f"{LABEL} shape"]) == [
        "time=210, spectral=72",
        "time=210, spectral=72",
        "time=100, spectral=72",
    ]
    assert table[f"{LABEL} rms"].iloc[0] == table[f"{LABEL} rms"].iloc[1]
    assert table[f"{LABEL} rms"].iloc[0] != table[f"{LABEL} rms"].iloc[2]
    assert table["cost"].iloc[0] == first.optimization_info.cost
    assert table["nfev"].iloc[0] == first.optimization_info.number_of_function_evaluations

    with pytest.warns(UserWarning, match="Skipped the damaged record"):
        assert list(project.list_results(source="a.ipynb").index) == [
            first.record.id,
            third.record.id,
        ]
    with pytest.warns(UserWarning, match="Skipped the damaged record"):
        assert list(project.list_results(name="target").index) == [second.record.id]
    with pytest.warns(UserWarning, match="Skipped the damaged record"):
        assert project.list_results(status="failed").empty


def test_list_results_of_empty_and_running(project: Project):
    """No results folder lists nothing; a record left at running is listed."""
    assert project.list_results().empty

    folder = project.results_folder / "2026-10-04_14-28-05"
    folder.mkdir(parents=True)
    record_module.write_record_file(
        folder,
        {"id": folder.name, "created": "2026-10-04T14:28:05+02:00", "status": "running"},
    )
    table = project.list_results()
    assert list(table["status"]) == ["running"]
    assert table["cost"].isna().all()


def test_compare_results(project: Project, scheme: Scheme, data: xr.Dataset):
    """Parameters, summary, scheme and data differences of two fits."""
    first = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    parallel_scheme_dict = SCHEME_DICT | {
        "experiments": {
            "parallel-decay": {
                "datasets": {
                    LABEL: SCHEME_DICT["experiments"][LABEL]["datasets"][LABEL]
                    | {"elements": ["parallel"]}
                }
            }
        }
    }
    second = project.optimize(
        Scheme.from_dict(parallel_scheme_dict), PARAMETERS, {LABEL: data * 2}, verbose=False
    )

    comparison = project.compare_results(first.record.id, second.record.path)

    columns = [first.record.id, second.record.id]
    assert list(comparison.parameters.columns) == columns
    assert ("rates.species_1", "value") in comparison.parameters.index
    assert ("rates.species_1", "vary") not in comparison.parameters.index
    assert comparison.summary.loc["cost"].tolist() == [
        first.optimization_info.cost,
        second.optimization_info.cost,
    ]
    assert f"{LABEL} weighted_root_mean_square_error" in comparison.summary.index
    assert re.search(r"^-\s+- sequential$", comparison.scheme_diff, flags=re.MULTILINE)
    assert re.search(r"^\+\s+- parallel$", comparison.scheme_diff, flags=re.MULTILINE)
    assert f"{LABEL}: data rms" in "\n".join(comparison.data_differences)
    assert "**Scheme**" in str(comparison)


def test_compare_result_with_its_record(project: Project, scheme: Scheme, data: xr.Dataset):
    """An in-memory result and its record show no differences."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)

    comparison = project.compare_results(result, result.record)

    assert comparison.parameters.empty
    assert comparison.scheme_diff == ""
    assert comparison.data_differences == []
    assert list(comparison.summary.columns) == ["a", result.record.id]
    statistics = comparison.summary.drop(index="converged")
    assert (statistics["a"] == statistics[result.record.id]).all()
