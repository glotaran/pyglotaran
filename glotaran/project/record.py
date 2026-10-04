"""Records of the fits run through a project.

A record is a folder with the minimum needed to recompute the results of a fit; it contains no
data and no result arrays.
"""

from __future__ import annotations

import os
import platform
import sys
from collections import Counter
from datetime import datetime
from importlib import metadata
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from warnings import warn

import numpy as np
import pandas as pd
import scipy
import xarray as xr

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import save_parameters
from glotaran.io import save_scheme
from glotaran.project.data_summary import data_source_path
from glotaran.project.data_summary import summarize_data
from glotaran.utils.io import relative_posix_path

if TYPE_CHECKING:
    from glotaran.optimization import Optimization
    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.project.scheme import Scheme

RECORD_FILE_NAME = "record.yml"
RECORD_SCHEMA_VERSION = 1
PLUGIN_ENTRY_POINT_GROUPS = (
    "glotaran.plugins.elements",
    "glotaran.plugins.data_io",
    "glotaran.plugins.project_io",
)


class RecordReference(NamedTuple):
    """Reference to the record of a fit."""

    id: str
    """Name of the record folder, the start time of the fit."""
    path: Path
    """Absolute path of the record folder."""


def detect_source() -> Path | None:
    """Detect the notebook or script this process runs.

    Notebooks are detected in VS Code (``__vsc_ipynb_file__``) and in Jupyter
    (``JPY_SESSION_NAME``), scripts by ``__main__.__file__``. Only an existing file counts.

    Returns
    -------
    Path | None
        Absolute path of the notebook or script, ``None`` if not detected (for example with
        nbclient, papermill, nbsphinx, Colab, Spyder or plain IPython).
    """
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    if shell is not None:
        candidates = [shell.user_ns.get("__vsc_ipynb_file__"), os.environ.get("JPY_SESSION_NAME")]
    else:
        candidates = [getattr(sys.modules.get("__main__"), "__file__", None)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def collect_environment() -> dict[str, str]:
    """Collect the versions of pyglotaran, its main dependencies and installed plugin packages.

    Returns
    -------
    dict[str, str]
    """
    # Prevent circular import
    from glotaran import __version__  # noqa: PLC0415

    environment = {
        "pyglotaran": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "xarray": xr.__version__,
    }
    entry_points = metadata.entry_points()
    for group in PLUGIN_ENTRY_POINT_GROUPS:
        for entry_point in entry_points.select(group=group):
            distribution = entry_point.dist
            if distribution is not None and distribution.name != "pyglotaran":
                environment[distribution.name] = distribution.version
    return environment


def claim_record_folder(results_folder: Path, created: datetime) -> Path:
    """Create the record folder named by the start time, with ``_2``, ``_3``, ... if taken.

    The folder is created with an exclusive ``mkdir``, so two fits that start in the same
    second, also in two processes sharing the results folder, get different folders.

    Parameters
    ----------
    results_folder : Path
        Folder the record folder is created in.
    created : datetime
        Start time of the fit.

    Returns
    -------
    Path
    """
    results_folder.mkdir(parents=True, exist_ok=True)
    name = created.strftime("%Y-%m-%d_%H-%M-%S")
    for number in count(1):
        folder = results_folder / (name if number == 1 else f"{name}_{number}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise AssertionError  # pragma: no cover


def write_record_file(folder: Path, content: dict[str, Any]) -> None:
    """Write ``record.yml`` atomically: to a temporary file that then replaces it.

    Parameters
    ----------
    folder : Path
        Record folder.
    content : dict[str, Any]
        Content of ``record.yml``.
    """
    temporary_file = folder / f"{RECORD_FILE_NAME}.tmp"
    write_dict(content, file_name=temporary_file)
    temporary_file.replace(folder / RECORD_FILE_NAME)


def read_record_file(folder: Path) -> dict[str, Any]:
    """Read ``record.yml``.

    Parameters
    ----------
    folder : Path
        Record folder.

    Returns
    -------
    dict[str, Any]

    Raises
    ------
    ValueError
        If the record has a newer major schema version than this version of pyglotaran reads.
    """
    content = dict(load_dict(folder / RECORD_FILE_NAME, is_file=True))
    if int(content.get("schema_version", 1)) > RECORD_SCHEMA_VERSION:
        msg = (
            f"The record in '{folder}' has schema version {content['schema_version']}; this "
            f"version of pyglotaran reads up to {RECORD_SCHEMA_VERSION}."
        )
        raise ValueError(msg)
    return content


def _float(value: float | None) -> float | None:
    return None if value is None else float(value)


def summarize_fit(result: Result, *, converged: bool | None = None) -> dict[str, Any]:
    """Summarize the statistics of a fit for its record.

    Parameters
    ----------
    result : Result
        Result of the fit.
    converged : bool | None
        SciPy's success flag, which the result does not hold.

    Returns
    -------
    dict[str, Any]
    """
    info = result.optimization_info
    return {
        "cost": _float(info.cost),
        "chi_square": _float(info.chi_square),
        "reduced_chi_square": _float(info.reduced_chi_square),
        "root_mean_square_error": _float(info.root_mean_square_error),
        "degrees_of_freedom": None
        if info.degrees_of_freedom is None
        else int(info.degrees_of_freedom),
        "number_of_function_evaluations": int(info.number_of_function_evaluations),
        "termination_reason": info.termination_reason,
        "converged": converged,
        "free_parameter_labels": list(info.free_parameter_labels),
        "datasets": {
            label: {
                "root_mean_square_error": _float(optimization_result.meta.root_mean_square_error),
                "weighted_root_mean_square_error": _float(
                    optimization_result.meta.weighted_root_mean_square_error
                ),
            }
            for label, optimization_result in result.optimization_results.items()
        },
    }


class FitRecord:
    """The record of one fit, written when the fit starts and completed when it ends.

    A failure to write the record gives a warning and never replaces the outcome of the fit.
    """

    def __init__(
        self, folder: Path, content: dict[str, Any], *, write_parameter_history: bool
    ) -> None:
        """Initialize a record; use :meth:`FitRecord.start` instead.

        Parameters
        ----------
        folder : Path
            Record folder.
        content : dict[str, Any]
            Content of ``record.yml``.
        write_parameter_history : bool
            Whether to write ``parameter_history.csv``.
        """
        self.folder = folder
        self.content = content
        self.write_parameter_history = write_parameter_history

    @property
    def reference(self) -> RecordReference:
        """Reference to this record."""
        return RecordReference(self.folder.name, self.folder)

    @classmethod
    def start(
        cls,
        results_folder: Path,
        scheme: Scheme,
        initial_parameters: Parameters,
        optimization: Optimization,
        *,
        name: str | None,
        write_parameter_history: bool,
    ) -> FitRecord | None:
        """Claim the record folder and write the parts known when the fit starts.

        Writes ``record.yml`` with status ``running``, ``scheme.yml`` and
        ``initial_parameters.csv``.

        Parameters
        ----------
        results_folder : Path
            Folder the record folder is created in.
        scheme : Scheme
            Scheme of the fit with the data loaded.
        initial_parameters : Parameters
            Initial parameters.
        optimization : Optimization
            The optimization, not yet run.
        name : str | None
            Optional label of the record.
        write_parameter_history : bool
            Whether to write ``parameter_history.csv`` when the fit ends.

        Returns
        -------
        FitRecord | None
            ``None`` if the record could not be written.
        """
        try:
            created = datetime.now().astimezone()
            folder = claim_record_folder(results_folder, created)
            source = detect_source()
            data_models = [
                (label, data_model)
                for experiment in scheme.experiments.values()
                for label, data_model in experiment.datasets.items()
            ]
            if max(Counter(label for label, _ in data_models).values()) > 1:
                warn(
                    "Dataset labels repeat across experiments, so the per-dataset entries of the "
                    f"record in '{folder}' are ambiguous.",
                    stacklevel=3,
                )
            content = {
                "schema_version": RECORD_SCHEMA_VERSION,
                "id": folder.name,
                "created": created.isoformat(timespec="seconds"),
                "status": "running",
                "error": None,
                "source": "unknown" if source is None else relative_posix_path(source, folder),
                "scheme_source": None
                if scheme.source_path is None
                else relative_posix_path(scheme.source_path, folder),
                "name": name,
                "environment": collect_environment(),
                "optimizer": optimization.settings.model_dump(),
                "data": {
                    label: {
                        "source_path": data_source_path(data_model.data, folder),
                        **summarize_data(data_model.data),
                    }
                    for label, data_model in data_models
                },
                "summary": None,
            }
            save_scheme(scheme, folder / "scheme.yml", update_source_path=False)
            save_parameters(
                initial_parameters, folder / "initial_parameters.csv", update_source_path=False
            )
            write_record_file(folder, content)
        except Exception as error:  # noqa: BLE001
            warn(f"The record of this fit could not be written: {error!r}", stacklevel=3)
            return None
        return cls(folder, content, write_parameter_history=write_parameter_history)

    def finish(
        self,
        optimization: Optimization,
        *,
        result: Result | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Write the outcome of the fit and replace ``record.yml``.

        Parameters
        ----------
        optimization : Optimization
            The optimization of the fit.
        result : Result | None
            Result of the fit, if it returned one.
        error : BaseException | None
            Exception that ended the fit, if any.
        """
        try:
            error = error or optimization.error
            if error is None:
                status = "success"
            elif isinstance(error, KeyboardInterrupt):
                status = "interrupted"
            else:
                status = "failed"
            cost_history = optimization.cost_history
            if status == "success" and result is not None:
                optimized_parameters: Parameters | None = result.optimized_parameters
                summary = summarize_fit(result, converged=optimization.converged)
            else:
                # The last evaluated parameters; after a crash in the objective, the ones that
                # raised. Without an evaluation they equal the initial parameters.
                optimized_parameters = None
                if cost_history:
                    # Without the standard errors they carry over from the initial parameters
                    optimized_parameters = optimization.parameters.copy()
                    for parameter in optimized_parameters.all():
                        parameter.standard_error = np.nan
                summary = {
                    "number_of_function_evaluations": len(cost_history),
                    "cost": cost_history[-1] if cost_history else None,
                }
            if optimized_parameters is not None:
                save_parameters(
                    optimized_parameters,
                    self.folder / "optimized_parameters.csv",
                    update_source_path=False,
                )
            pd.DataFrame(
                {"evaluation": np.arange(1, len(cost_history) + 1), "cost": cost_history}
            ).to_csv(self.folder / "cost_history.csv", index=False)
            if self.write_parameter_history:
                optimization.parameter_history.to_dataframe().to_csv(
                    self.folder / "parameter_history.csv", index=False
                )
            self.content |= {
                "status": status,
                "error": None
                if error is None
                else {"type": type(error).__name__, "message": str(error)},
                "summary": summary,
            }
            write_record_file(self.folder, self.content)
        except Exception as write_error:  # noqa: BLE001
            warn(
                f"The record of this fit in '{self.folder}' could not be completed: "
                f"{write_error!r}",
                stacklevel=3,
            )
