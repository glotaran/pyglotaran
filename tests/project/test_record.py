from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.io import save_dataset
from glotaran.io import save_scheme
from glotaran.optimization.objective import OptimizationObjective
from glotaran.project import Project
from glotaran.project import record as record_module
from glotaran.project.record import claim_record_folder
from glotaran.project.record import detect_source
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from tests.project.conftest import LABEL
from tests.project.conftest import scheme_dict_with_data

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from glotaran.project import Scheme

CREATED = datetime(2026, 10, 4, 14, 28, 5, tzinfo=timezone.utc)


def read_record(folder: Path) -> dict:
    return load_dict(folder / "record.yml", is_file=True)


def only_record(project: Project) -> Path:
    (folder,) = project.results_folder.iterdir()
    return folder


def raise_from_evaluation(
    monkeypatch: pytest.MonkeyPatch, number: int, error: BaseException
) -> None:
    """Make the objective raise ``error`` from its ``number``-th evaluation on."""
    calculate = OptimizationObjective.calculate
    calls = []

    def failing_calculate(self: OptimizationObjective) -> np.ndarray:
        calls.append(None)
        if len(calls) >= number:
            raise error
        return calculate(self)

    monkeypatch.setattr(OptimizationObjective, "calculate", failing_calculate)


def test_record_of_a_fit(project: Project, scheme: Scheme, data: xr.Dataset):
    """A fit writes a record with metadata, parameters and histories, and no arrays."""
    result = project.optimize(scheme, PARAMETERS, {LABEL: data}, name="first try")

    folder = only_record(project)
    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", folder.name)
    assert result.record == (folder.name, folder)
    assert sorted(path.name for path in folder.iterdir()) == [
        "cost_history.csv",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "parameter_history.csv",
        "record.yml",
        "scheme.yml",
    ]

    record = read_record(folder)
    assert record["schema_version"] == 1
    assert record["id"] == folder.name
    assert datetime.fromisoformat(record["created"]).utcoffset() is not None
    assert record["status"] == "success"
    assert record["error"] is None
    assert record["name"] == "first try"
    assert record["scheme_source"] is None
    assert set(record["environment"]) >= {"pyglotaran", "python", "numpy", "scipy", "xarray"}
    assert record["optimizer"]["optimization_method"] == "TrustRegionReflection"
    assert record["data"][LABEL]["source_path"] is None
    assert record["data"][LABEL]["shape"] == {"time": 210, "spectral": 72}
    summary = record["summary"]
    info = result.optimization_info
    assert summary["cost"] == info.cost
    assert summary["root_mean_square_error"] == info.root_mean_square_error
    assert summary["number_of_function_evaluations"] == info.number_of_function_evaluations
    assert summary["converged"] is True
    assert summary["free_parameter_labels"] == info.free_parameter_labels
    assert summary["datasets"][LABEL]["root_mean_square_error"] == pytest.approx(
        result.optimization_results[LABEL].meta.root_mean_square_error, rel=1e-15
    )

    cost_history = pd.read_csv(folder / "cost_history.csv")
    assert list(cost_history.columns) == ["evaluation", "cost"]
    assert len(cost_history) > info.number_of_function_evaluations
    parameter_history = pd.read_csv(folder / "parameter_history.csv")
    assert len(parameter_history) == len(cost_history) + 1

    optimized_parameters = load_parameters(folder / "optimized_parameters.csv")
    assert optimized_parameters.close_or_equal(result.optimized_parameters, rtol=1e-15)
    assert load_parameters(folder / "initial_parameters.csv") == PARAMETERS


def test_parameter_history_only_with_verbose(project: Project, scheme: Scheme, data: xr.Dataset):
    project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    assert not (only_record(project) / "parameter_history.csv").exists()


def test_scheme_and_data_source_paths(project: Project, scheme: Scheme, data: xr.Dataset):
    """Scheme file and data file are referenced relative to the record folder."""
    save_scheme(scheme, project.folder / "models" / "scheme.yml")
    save_dataset(data, project.folder / "data" / "data.nc")
    project.optimize(
        load_scheme(project.folder / "models" / "scheme.yml"),
        PARAMETERS,
        {LABEL: load_dataset(project.folder / "data" / "data.nc")},
        verbose=False,
    )

    record = read_record(only_record(project))
    assert record["scheme_source"] == "../../models/scheme.yml"
    assert record["data"][LABEL]["source_path"] == "../../data/data.nc"


def test_scheme_file_copied_unless_it_names_data_files(
    project: Project, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """A scheme file is recorded verbatim, with its comments, unless it names data files."""
    monkeypatch.chdir(project.folder)
    save_dataset(data, "data.nc")
    text = f"# First model\n{write_dict(SCHEME_DICT)}"
    (project.folder / "scheme.yml").write_text(text, encoding="utf8")
    write_dict(scheme_dict_with_data("data.nc"), file_name=project.folder / "with_data.yml")

    for name in ("scheme.yml", "with_data.yml"):
        scheme = load_scheme(project.folder / name)
        project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)

    plain, with_data = sorted(project.results_folder.iterdir())
    assert (plain / "scheme.yml").read_text(encoding="utf8") == text
    recorded = load_dict(with_data / "scheme.yml", is_file=True)
    assert "data" not in recorded["experiments"][LABEL]["datasets"][LABEL]


def test_plain_optimize_and_dry_run_write_nothing(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(project.folder)
    result = scheme.optimize(PARAMETERS, {LABEL: data}, verbose=False)
    dry_run = project.optimize(scheme, PARAMETERS, {LABEL: data}, dry_run=True, verbose=False)

    assert result.record is None
    assert dry_run.record is None
    assert sorted(path.name for path in project.folder.iterdir()) == ["project.gta"]


def test_claim_record_folder_suffix(tmp_path: Path):
    """Fits that start in the same second get ``_2``, ``_3``, ..."""
    names = [claim_record_folder(tmp_path, CREATED).name for _ in range(3)]
    assert names == ["2026-10-04_14-28-05", "2026-10-04_14-28-05_2", "2026-10-04_14-28-05_3"]


def test_claim_record_folder_in_concurrent_processes(tmp_path: Path):
    """Two processes sharing a results folder claim different folders for the same second."""
    script = (
        "import sys, datetime, pathlib; "
        "from glotaran.project.record import claim_record_folder; "
        "created = datetime.datetime(2026, 10, 4, 14, 28, 5); "
        "[claim_record_folder(pathlib.Path(sys.argv[1]), created) for _ in range(5)]"
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])  # noqa: S603
        for _ in range(2)
    ]
    assert [process.wait(timeout=120) for process in processes] == [0, 0]
    assert len(list(tmp_path.iterdir())) == 10


def test_unwritable_results_folder_warns(tmp_path: Path, scheme: Scheme, data: xr.Dataset):
    """A record that cannot be written gives a warning; the fit still returns its result."""
    (tmp_path / "results").write_text("a file blocks the results folder")
    project = Project.start(tmp_path)

    with pytest.warns(UserWarning, match="record of this fit could not be written"):
        result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    assert result.record is None
    assert result.optimization_info.success


def test_failed_fit_is_recorded(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """A fit whose objective raises is recorded with the parameters that raised.

    They hold no standard errors, also when the initial parameters have some.
    """
    parameters = PARAMETERS.copy()
    for parameter in parameters.all():
        parameter.standard_error = 0.1
    raise_from_evaluation(monkeypatch, 4, ValueError("objective failed"))

    with pytest.warns(UserWarning, match="Optimization failed"), pytest.raises(ValueError):
        project.optimize(scheme, parameters, {LABEL: data}, verbose=False)

    folder = only_record(project)
    record = read_record(folder)
    assert record["status"] == "failed"
    assert record["error"] == {"type": "ValueError", "message": "objective failed"}
    cost_history = pd.read_csv(folder / "cost_history.csv")
    assert record["summary"] == {
        "number_of_function_evaluations": 3,
        "cost": cost_history["cost"].iloc[-1],
    }
    optimized_parameters = load_parameters(folder / "optimized_parameters.csv")
    assert all(np.isnan(parameter.standard_error) for parameter in optimized_parameters.all())


def test_failed_fit_with_result_is_recorded(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """With ``raise_exception=False`` an optimizer error returns a result and a failed record."""

    def failing_least_squares(fun, x0, **kwargs):  # noqa: ANN001, ANN003
        fun(x0)
        fun(x0 * 1.01)
        msg = "optimizer failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("glotaran.optimization.optimization.least_squares", failing_least_squares)
    with pytest.warns(UserWarning, match="Optimization failed"):
        result = project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)

    record = read_record(result.record.path)
    assert record["status"] == "failed"
    assert record["error"] == {"type": "RuntimeError", "message": "optimizer failed"}
    assert record["summary"]["number_of_function_evaluations"] == 2
    assert result.optimization_info.number_of_function_evaluations == 2
    optimized_parameters = load_parameters(result.record.path / "optimized_parameters.csv")
    assert optimized_parameters.close_or_equal(result.optimized_parameters, rtol=1e-15)


def test_interrupted_fit_is_recorded(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    raise_from_evaluation(monkeypatch, 3, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)

    record = read_record(only_record(project))
    assert record["status"] == "interrupted"
    assert record["error"]["type"] == "KeyboardInterrupt"
    assert record["summary"]["number_of_function_evaluations"] == 2


def test_record_write_failure_keeps_the_fit_error(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    """An error while completing the record warns and does not replace the error of the fit."""
    raise_from_evaluation(monkeypatch, 1, ValueError("objective failed"))
    write_record_file = record_module.write_record_file
    calls = []

    def failing_second_write(folder: Path, content: dict) -> None:
        calls.append(None)
        if len(calls) > 1:
            msg = "disk full"
            raise OSError(msg)
        write_record_file(folder, content)

    monkeypatch.setattr(record_module, "write_record_file", failing_second_write)

    with (
        pytest.warns(UserWarning, match="could not be completed"),
        pytest.warns(UserWarning, match="Optimization failed"),
        pytest.raises(ValueError, match="objective failed"),
    ):
        project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    assert read_record(only_record(project))["status"] == "running"


def fake_ipython(monkeypatch: pytest.MonkeyPatch, user_ns: dict | None) -> None:
    module = ModuleType("IPython")
    module.get_ipython = lambda: None if user_ns is None else SimpleNamespace(user_ns=user_ns)  # type:ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", module)


def test_detect_source_vscode_notebook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    notebook = tmp_path / "analysis.ipynb"
    notebook.touch()
    fake_ipython(monkeypatch, {"__vsc_ipynb_file__": str(notebook)})
    assert detect_source() == notebook.resolve()


def test_detect_source_jupyter_notebook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    notebook = tmp_path / "analysis.ipynb"
    notebook.touch()
    fake_ipython(monkeypatch, {})
    monkeypatch.setenv("JPY_SESSION_NAME", str(notebook))
    assert detect_source() == notebook.resolve()


def test_detect_source_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "fit.py"
    script.touch()
    fake_ipython(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__file__=str(script)))
    assert detect_source() == script.resolve()


def test_detect_source_unknown(monkeypatch: pytest.MonkeyPatch):
    """In IPython without a known notebook variable the source is unknown."""
    fake_ipython(monkeypatch, {})
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    assert detect_source() is None
    monkeypatch.setenv("JPY_SESSION_NAME", "renamed_or_relative.ipynb")
    assert detect_source() is None


def test_recorded_source_is_relative(
    project: Project, scheme: Scheme, data: xr.Dataset, monkeypatch: pytest.MonkeyPatch
):
    script = project.folder / "fit.py"
    script.touch()
    fake_ipython(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__file__=str(script)))

    project.optimize(scheme, PARAMETERS, {LABEL: data}, verbose=False)
    assert read_record(only_record(project))["source"] == "../../fit.py"
