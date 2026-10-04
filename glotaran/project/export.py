"""Export of a result as a self-contained folder."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from warnings import warn

import xarray as xr

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.project.data_summary import summarize_data
from glotaran.project.record import RECORD_SCHEMA_VERSION
from glotaran.project.record import collect_environment
from glotaran.project.record import detect_source
from glotaran.project.record import read_record_file
from glotaran.project.record import summarize_fit

if TYPE_CHECKING:
    from glotaran.io.interface import SavingOptions
    from glotaran.project.result import Result

EXPORT_FILE_NAME = "export.yml"


def export_result(
    result: Result,
    folder: Path,
    *,
    project_file: Path,
    overwrite: bool,
    saving_options: SavingOptions,
    include_source_files: bool,
) -> Path:
    """Export a result to ``folder``; see :meth:`Project.export`.

    The export is written to a temporary folder next to ``folder`` and renamed when complete,
    so a failed export leaves no partial folder.

    Parameters
    ----------
    result : Result
        The result.
    folder : Path
        Export folder.
    project_file : Path
        ``project.gta`` of the project, copied verbatim if it exists.
    overwrite : bool
        Replace an existing export.
    saving_options : SavingOptions
        Saving options of :meth:`Result.save`.
    include_source_files : bool
        Copy the files the datasets were loaded from.

    Returns
    -------
    Path
        The export folder.

    Raises
    ------
    FileExistsError
        If ``overwrite`` is ``True`` and ``folder`` exists but contains no export.
    """
    if folder.exists():
        if not overwrite:
            warn(
                f"An export named '{folder.name}' already exists in '{folder.parent}' and may be "
                "stale; nothing was written. Pass overwrite=True to replace it.",
                stacklevel=3,
            )
            return folder
        if not (folder / EXPORT_FILE_NAME).is_file():
            msg = f"'{folder}' exists and is not an export; it is not overwritten."
            raise FileExistsError(msg)

    changed_parameters = [
        parameter.label
        for parameter in result.optimized_parameters.all()
        if not result._fitted_parameters.has(parameter.label)  # noqa: SLF001
        or not parameter._deep_equals(result._fitted_parameters.get(parameter.label))  # noqa: SLF001
    ]
    if changed_parameters:
        warn(
            "The optimized parameters were changed after the fit; the export contains the "
            f"current values of: {', '.join(changed_parameters)}.",
            stacklevel=3,
        )

    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary_folder = folder.with_name(f".{folder.name}.{os.getpid()}.tmp")
    # Input data are always written as data: with ``input_data`` in the data filter,
    # ``Result.save`` would write a reference to the file the data was loaded from instead.
    data_filter = set(saving_options.get("data_filter", set())) - {"input_data"}
    # Before saving, which sets the source path of the scheme to the saved file
    metadata = export_metadata(result, changed_parameters)
    try:
        result.save(
            temporary_folder,
            format_name="yml",
            saving_options=saving_options | {"data_filter": data_filter},
        )
        if project_file.is_file():
            shutil.copyfile(project_file, temporary_folder / project_file.name)
        if include_source_files:
            metadata["source_files"] = copy_source_files(result, temporary_folder)
        write_dict(metadata, file_name=temporary_folder / EXPORT_FILE_NAME)
    except BaseException:
        shutil.rmtree(temporary_folder, ignore_errors=True)
        raise
    if folder.exists():
        shutil.rmtree(folder)
    temporary_folder.rename(folder)
    return folder


def export_metadata(result: Result, changed_parameters: list[str]) -> dict[str, Any]:
    """Create the content of ``export.yml``, without the copied source files.

    The metadata of the record of the result are used where available, else they are
    detected now. Paths are reduced to file names, so that a published export contains no
    local paths.

    Parameters
    ----------
    result : Result
        The exported result.
    changed_parameters : list[str]
        Labels of optimized parameters changed after the fit.

    Returns
    -------
    dict[str, Any]
    """
    if result.record is not None and (result.record.path / "record.yml").is_file():
        record = read_record_file(result.record.path)
        source = record.get("source") or "unknown"
        scheme_source = record.get("scheme_source")
    else:
        record = {}
        source = detect_source() or "unknown"
        scheme_source = result.scheme.source_path
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "exported": datetime.now().astimezone().isoformat(timespec="seconds"),
        "record_id": None if result.record is None else result.record.id,
        "source": Path(source).name,
        "scheme_source": None if scheme_source is None else Path(scheme_source).name,
        "name": record.get("name"),
        "environment": record.get("environment") or collect_environment(),
        "optimizer": None
        if result.optimizer_settings is None
        else result.optimizer_settings.model_dump(),
        "data": {
            label: summarize_data(
                input_data.to_dataset(name="data")
                if isinstance(input_data, xr.DataArray)
                else input_data
            )
            for label, input_data in result.input_data.items()
        },
        "summary": summarize_fit(result, converged=(record.get("summary") or {}).get("converged")),
        "changed_parameters": changed_parameters,
        "source_files": {},
    }


def copy_source_files(result: Result, folder: Path) -> dict[str, str | None]:
    """Copy the files the datasets were loaded from to ``source_files/<dataset label>/``.

    Best effort: a file that is missing or cannot be read gives a warning. The files are not
    compared with the input data, from which they can differ.

    Parameters
    ----------
    result : Result
        The exported result.
    folder : Path
        Export folder.

    Returns
    -------
    dict[str, str | None]
        Per dataset loaded from a file the copied file relative to ``folder``, or ``None`` if
        it could not be copied.
    """
    copied: dict[str, str | None] = {}
    for label, input_data in result.input_data.items():
        if "io_plugin_name" not in input_data.attrs or "source_path" not in input_data.attrs:
            continue
        source_file = Path(input_data.attrs["source_path"]).resolve()
        target = folder / "source_files" / label / source_file.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        except OSError as error:
            warn(f"The source file of '{label}' could not be copied: {error!r}", stacklevel=4)
            copied[label] = None
        else:
            copied[label] = target.relative_to(folder).as_posix()
    return copied


def read_export_files(folder: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read ``export.yml`` and the file names of scheme and parameters from ``result.yml``.

    Parameters
    ----------
    folder : Path
        Export folder.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        The export metadata, and the files of ``scheme``, ``initial_parameters`` and
        ``optimized_parameters`` relative to ``folder``.
    """
    metadata = dict(load_dict(folder / EXPORT_FILE_NAME, is_file=True))
    result_spec = load_dict(folder / "result.yml", is_file=True)
    files = {
        key: result_spec[key] for key in ("scheme", "initial_parameters", "optimized_parameters")
    }
    return metadata, files
