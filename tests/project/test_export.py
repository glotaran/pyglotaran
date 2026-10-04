from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import load_dataset
from glotaran.io import load_result
from glotaran.io import load_scheme
from glotaran.io import save_dataset
from glotaran.project import Project
from glotaran.project import Scheme
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from tests.project.conftest import LABEL
from tests.project.conftest import scheme_dict_with_data
from tests.project.test_record import fake_ipython

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr


def tree(folder: Path) -> dict[str, float]:
    return {
        path.relative_to(folder).as_posix(): path.stat().st_mtime_ns
        for path in folder.rglob("*")
        if path.is_file()
    }


def test_export_loads_without_the_original_files(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """An export with the default saving options is self-contained."""
    project.project_file.write_text("title: Sequential decay\n", encoding="utf8")
    save_dataset(data, project.folder / "data.nc")
    script = project.folder / "analysis.py"
    script.touch()
    fake_ipython(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__file__=str(script)))
    result = project.optimize(
        scheme, PARAMETERS, {LABEL: load_dataset(project.folder / "data.nc")}, name="first"
    )

    folder = project.export(result)

    assert folder == project.exports_folder / "last_result"
    assert (folder / "project.gta").read_bytes() == project.project_file.read_bytes()
    metadata = load_dict(folder / "export.yml", is_file=True)
    assert metadata["record_id"] == result.record.id
    assert metadata["source"] == "analysis.py"
    assert metadata["scheme_source"] is None
    assert metadata["name"] == "first"
    assert metadata["summary"]["cost"] == result.optimization_info.cost
    assert metadata["summary"]["converged"] is True
    assert metadata["data"][LABEL]["shape"] == {"time": 210, "spectral": 72}
    assert metadata["changed_parameters"] == []
    assert not list(project.exports_folder.glob(".*"))

    (project.folder / "data.nc").rename(project.folder / "moved.nc")
    loaded = load_result(folder)
    assert loaded.input_data[LABEL]["data"].equals(result.input_data[LABEL])
    assert loaded.optimization_results[LABEL].fitted_data.equals(
        result.optimization_results[LABEL].fitted_data
    )
    assert loaded.optimized_parameters == result.optimized_parameters


def test_export_writes_preprocessed_input_data(project: Project, scheme: Scheme, data: xr.Dataset):
    """Data loaded from a file and preprocessed are exported as fitted, also when filtered."""
    save_dataset(data, project.folder / "raw.nc")
    preprocessed = load_dataset(project.folder / "raw.nc").isel(time=slice(0, 100)) * 1000
    preprocessed.attrs = load_dataset(project.folder / "raw.nc").attrs
    result = project.optimize(scheme, PARAMETERS, {LABEL: preprocessed}, verbose=False)

    folder = project.export(result, "minimal", saving_options=SAVING_OPTIONS_MINIMAL)

    assert not (folder / "optimization_results" / LABEL / "residuals.nc").exists()
    loaded = load_result(folder)
    assert loaded.input_data[LABEL]["data"].shape == (100, 72)
    assert loaded.input_data[LABEL]["data"].equals(result.input_data[LABEL])


def test_source_files(project: Project, scheme: Scheme, data: xr.Dataset):
    """Source files are copied on request; a missing one warns and the export succeeds."""
    save_dataset(data, project.folder / "raw.nc")
    result = project.optimize(
        scheme, PARAMETERS, {LABEL: load_dataset(project.folder / "raw.nc")}, verbose=False
    )

    folder = project.export(result, "with_source", include_source_files=True)
    assert (folder / "source_files" / LABEL / "raw.nc").is_file()
    metadata = load_dict(folder / "export.yml", is_file=True)
    assert metadata["source_files"] == {LABEL: f"source_files/{LABEL}/raw.nc"}

    (project.folder / "raw.nc").unlink()
    with pytest.warns(UserWarning, match=f"source file of '{LABEL}' could not be copied"):
        folder = project.export(result, "missing_source", include_source_files=True)
    assert load_dict(folder / "export.yml", is_file=True)["source_files"] == {LABEL: None}
    assert load_result(folder).input_data[LABEL]["data"].equals(result.input_data[LABEL])


def test_existing_export(project: Project, scheme: Scheme, data: xr.Dataset, tmp_path: Path):
    """An existing export is kept with a warning unless overwritten; other folders never."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    folder = project.export(result, "paper_fig3")
    before = tree(folder)

    with pytest.warns(UserWarning, match="already exists .* may be stale; nothing was written"):
        assert project.export(result, "paper_fig3") == folder
    assert tree(folder) == before

    (folder / "notes.txt").write_text("not part of the export")
    project.export(result, "paper_fig3", overwrite=True)
    assert not (folder / "notes.txt").exists()
    assert set(tree(folder)) == set(before)

    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="is not an export"):
        project.export(result, tmp_path / "other", overwrite=True)
    assert (tmp_path / "other" / "keep.txt").is_file()


def test_export_keeps_the_result_source_path(project: Project, scheme: Scheme, data: xr.Dataset):
    """Export does not set ``Result.source_path`` and writes no local path to result.yml."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    folder = project.export(result)

    assert result.source_path is None

    loaded = load_result(folder)
    second = project.export(loaded, "second")

    assert loaded.source_path == folder
    assert "source_path" not in load_dict(second / "result.yml", is_file=True)


def test_folder_created_during_export_is_kept(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """A folder created by someone else while the export is written is not replaced."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    folder = project.exports_folder / "paper_fig3"
    save = type(result).save

    def save_and_create_folder(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        save(self, *args, **kwargs)
        folder.mkdir(parents=True)
        (folder / "keep.txt").write_text("keep")

    monkeypatch.setattr(type(result), "save", save_and_create_folder)
    with pytest.raises(FileExistsError, match="was created during the export"):
        project.export(result, "paper_fig3")
    assert [path.name for path in folder.iterdir()] == ["keep.txt"]
    assert [path.name for path in folder.parent.iterdir()] == ["paper_fig3"]


def test_overlapping_exports_to_one_name(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """An export started while another one to the same name is written keeps its own files."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    save = type(result).save
    calls = []

    def save_and_export_again(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        save(self, *args, **kwargs)
        calls.append(None)
        if len(calls) == 1:
            project.export(result, "paper_fig3")

    monkeypatch.setattr(type(result), "save", save_and_export_again)
    with pytest.raises(FileExistsError, match="was created during the export"):
        project.export(result, "paper_fig3")

    assert [path.name for path in project.exports_folder.iterdir()] == ["paper_fig3"]
    loaded = load_result(project.exports_folder / "paper_fig3")
    assert loaded.scheme.model_dump() == result.scheme.model_dump()


def test_scheme_file_with_data_paths(
    project: Project, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """Record and export of a scheme file that names data files work without those files."""
    monkeypatch.chdir(project.folder)
    save_dataset(data, "data.nc")
    write_dict(scheme_dict_with_data("data.nc"), file_name=project.folder / "scheme.yml")
    scheme = load_scheme(project.folder / "scheme.yml")
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    folder = project.export(result)

    (project.folder / "data.nc").unlink()

    assert load_result(folder).scheme.model_dump() == result.scheme.model_dump()
    project.recompute(result.record, {LABEL: data})
    project.recompute(folder, {LABEL: data})


def test_export_parameters_changed_in_place(project: Project, scheme: Scheme, data: xr.Dataset):
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    result.optimized_parameters.get("rates.species_1").value = 0.4

    with pytest.warns(UserWarning, match="changed after the fit.*rates.species_1"):
        folder = project.export(result)

    assert load_dict(folder / "export.yml", is_file=True)["changed_parameters"] == [
        "rates.species_1"
    ]
    assert load_result(folder).optimized_parameters.get("rates.species_1").value == 0.4


def test_export_of_a_result_without_record(scheme: Scheme, data: xr.Dataset, tmp_path: Path):
    """A result of ``scheme.optimize`` exports with metadata detected at export time."""
    project = Project.start(tmp_path)
    result = scheme.optimize(PARAMETERS, {LABEL: data}, verbose=False)

    folder = project.export(result)

    metadata = load_dict(folder / "export.yml", is_file=True)
    assert metadata["record_id"] is None
    assert metadata["summary"]["converged"] is None
    assert not project.results_folder.exists()


def test_recompute_and_compare_an_export(project: Project, scheme: Scheme, data: xr.Dataset):
    """An export can be recomputed and compared like a record."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    folder = project.export(result)

    recomputed = project.recompute(folder, {LABEL: data})
    assert recomputed.optimization_results[LABEL].fitted_data.equals(
        result.optimization_results[LABEL].fitted_data
    )
    assert recomputed.recomputation["original_fit"]["id"] == result.record.id
    assert recomputed.record is None

    comparison = project.compare_results(result.record, folder)
    assert comparison.parameters.empty
    assert comparison.scheme_diff == ""
    assert comparison.data_differences == []


def test_export_of_a_recomputed_result(project: Project, scheme: Scheme, data: xr.Dataset):
    """A recomputed result exports with its original-fit and reconstruction information."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    recomputed = project.recompute(result.record, {LABEL: data})

    folder = project.export(recomputed, os.fspath(project.folder / "elsewhere"))

    assert folder == project.folder / "elsewhere"
    metadata = load_dict(folder / "export.yml", is_file=True)
    assert metadata["record_id"] == result.record.id
    assert metadata["recomputed_from"] == result.record.id
    # The summary of the original fit, not that of the recompute (a dry run)
    record_summary = load_dict(result.record.path / "record.yml", is_file=True)["summary"]
    assert metadata["summary"] == record_summary
    loaded = load_result(folder)
    assert loaded.recomputation["original_fit"]["id"] == result.record.id
    assert loaded.recomputation["reconstruction"]["data_differences"] == []
    # A recompute of the export keeps the original fit's summary
    again = project.recompute(folder, {LABEL: data})
    assert again.recomputation["original_fit"]["summary"] == record_summary
