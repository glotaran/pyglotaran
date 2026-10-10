"""Listing and comparison of recorded fits."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from warnings import warn

import pandas as pd
import xarray as xr

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.parameter import Parameters
from glotaran.project.data_summary import compare_data_summaries
from glotaran.project.data_summary import summarize_data
from glotaran.project.record import RECORD_FILE_NAME
from glotaran.project.record import read_record_file
from glotaran.project.record import summarize_fit

if TYPE_CHECKING:
    from pathlib import Path

    from glotaran.project.result import Result
    from glotaran.project.scheme import Scheme

PARAMETER_FIELDS = (
    "value",
    "standard_error",
    "expression",
    "minimum",
    "maximum",
    "non_negative",
    "vary",
)
SUMMARY_STATISTICS = (
    "cost",
    "chi_square",
    "reduced_chi_square",
    "root_mean_square_error",
    "degrees_of_freedom",
    "number_of_function_evaluations",
    "termination_reason",
    "converged",
)


def list_records(
    results_folder: Path,
    *,
    source: str | None = None,
    scheme_source: str | None = None,
    name: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """List the records in a results folder, sorted by start time.

    Only folders with a ``record.yml`` are listed; a damaged ``record.yml`` is skipped with a
    warning.

    Parameters
    ----------
    results_folder : Path
        Folder with the records.
    source : str | None
        Only records whose ``source`` contains this text.
    scheme_source : str | None
        Only records whose ``scheme_source`` contains this text.
    name : str | None
        Only records whose ``name`` contains this text.
    status : str | None
        Only records with this status (``running``, ``success``, ``failed`` or ``interrupted``).

    Returns
    -------
    pd.DataFrame
        One row per record, indexed by ``id``.
    """
    records = []
    folders = sorted(results_folder.iterdir()) if results_folder.is_dir() else []
    for folder in folders:
        if not (folder / RECORD_FILE_NAME).is_file():
            continue
        try:
            record = read_record_file(folder)
            records.append(
                (datetime.fromisoformat(str(record["created"])), record, _list_row(record))
            )
        except Exception as error:  # noqa: BLE001
            warn(f"Skipped the damaged record '{folder}': {error!r}", stacklevel=3)
    filters = {"source": source, "scheme_source": scheme_source, "name": name}
    rows = [
        row
        for _, record, row in sorted(records, key=lambda entry: entry[0])
        if all(text is None or text in str(record.get(key) or "") for key, text in filters.items())
        and (status is None or record.get("status") == status)
    ]
    return pd.DataFrame(rows, columns=None if rows else ["id"]).set_index("id")


def _list_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: record.get(key)
        for key in ("id", "created", "source", "name", "scheme_source", "status")
    }
    for label, entry in (record.get("data") or {}).items():
        shape = entry.get("shape") or {}
        row[f"{label} shape"] = ", ".join(f"{dim}={size}" for dim, size in shape.items())
        row[f"{label} rms"] = (entry.get("data") or {}).get("rms")
    summary = record.get("summary") or {}
    row["cost"] = summary.get("cost")
    row["rmse"] = summary.get("root_mean_square_error")
    row["nfev"] = summary.get("number_of_function_evaluations")
    return row


class Fit(NamedTuple):
    """What a comparison needs of a fit."""

    label: str
    parameters: Parameters
    summary: dict[str, Any]
    scheme_text: str
    data: dict[str, dict[str, Any]]


def fit_from_record(folder: Path) -> Fit:
    """Read the fit of a record folder for a comparison.

    Parameters
    ----------
    folder : Path
        Record folder.

    Returns
    -------
    Fit
    """
    content = read_record_file(folder)
    parameters_file = folder / "optimized_parameters.csv"
    return Fit(
        label=str(content.get("id", folder.name)),
        parameters=load_parameters(parameters_file)
        if parameters_file.is_file()
        else Parameters.empty(),
        summary=content.get("summary") or {},
        scheme_text=scheme_text(load_scheme(folder / "scheme.yml")),
        data=content.get("data") or {},
    )


def fit_from_result(result: Result, label: str) -> Fit:
    """Describe an in-memory result for a comparison.

    Parameters
    ----------
    result : Result
        The result.
    label : str
        Label of the fit in the comparison.

    Returns
    -------
    Fit
    """
    return Fit(
        label=label,
        parameters=result.optimized_parameters,
        summary=summarize_fit(result),
        scheme_text=scheme_text(result.scheme),
        data={
            dataset_label: summarize_data(
                input_data.to_dataset(name="data")
                if isinstance(input_data, xr.DataArray)
                else input_data
            )
            for dataset_label, input_data in result.input_data.items()
        },
    )


def scheme_text(scheme: Scheme) -> str:
    """Write a scheme as YAML text in a normalized form, so that only content differences show.

    Parameters
    ----------
    scheme : Scheme
        The scheme.

    Returns
    -------
    str
    """
    return write_dict(scheme.model_dump(exclude_unset=True, mode="json"))  # type:ignore[return-value]


@dataclass
class FitComparison:
    """Differences between two fits."""

    parameters: pd.DataFrame
    """Parameter fields that differ, indexed by parameter label and field."""
    summary: pd.DataFrame
    """Summary statistics of both fits, including the RMSE per dataset."""
    scheme_diff: str
    """Unified diff of the schemes; empty if they are equal."""
    data_differences: list[str]
    """What changed in the data summaries per dataset and by how much."""

    def _repr_markdown_(self) -> str:
        """Render the comparison as markdown."""
        data = "\n".join(f"- {difference}" for difference in self.data_differences)
        return "\n\n".join(
            [
                "**Parameters**",
                self.parameters.to_markdown(floatfmt=".10g")
                if len(self.parameters)
                else "No differences.",
                "**Summary**",
                self.summary.to_markdown(floatfmt=".10g"),
                "**Scheme**",
                f"```diff\n{self.scheme_diff}```" if self.scheme_diff else "No differences.",
                "**Data**",
                data or "No differences.",
            ]
        )

    def __str__(self) -> str:
        """Render the comparison as markdown."""
        return self._repr_markdown_()


def compare_fits(a: Fit, b: Fit) -> FitComparison:
    """Compare two fits.

    Parameters
    ----------
    a : Fit
        First fit.
    b : Fit
        Second fit.

    Returns
    -------
    FitComparison
    """
    columns = [a.label, b.label] if a.label != b.label else ["a", "b"]
    parameter_rows = {}
    for label in sorted(set(a.parameters.labels) | set(b.parameters.labels)):
        for field in PARAMETER_FIELDS:
            values = [
                getattr(fit.parameters.get(label), field) if fit.parameters.has(label) else None
                for fit in (a, b)
            ]
            if not _same(*values):
                parameter_rows[(label, field)] = values
    parameters = pd.DataFrame.from_dict(parameter_rows, orient="index", columns=columns)
    if len(parameters):
        parameters.index = pd.MultiIndex.from_tuples(parameters.index, names=["label", "field"])

    summary_rows = {
        statistic: [fit.summary.get(statistic) for fit in (a, b)]
        for statistic in SUMMARY_STATISTICS
    }
    for label in dict.fromkeys([*a.data, *b.data]):
        for key in ("root_mean_square_error", "weighted_root_mean_square_error"):
            summary_rows[f"{label} {key}"] = [
                ((fit.summary.get("datasets") or {}).get(label) or {}).get(key) for fit in (a, b)
            ]
    summary = pd.DataFrame.from_dict(summary_rows, orient="index", columns=columns)

    scheme_diff = "".join(
        difflib.unified_diff(
            a.scheme_text.splitlines(keepends=True),
            b.scheme_text.splitlines(keepends=True),
            fromfile=columns[0],
            tofile=columns[1],
        )
    )
    data_differences = []
    for label in dict.fromkeys([*a.data, *b.data]):
        if label not in b.data:
            data_differences.append(f"{label}: only in {columns[0]}")
        elif label not in a.data:
            data_differences.append(f"{label}: only in {columns[1]}")
        else:
            data_differences += compare_data_summaries(label, a.data[label], b.data[label])
    return FitComparison(parameters, summary, scheme_diff, data_differences)


def _same(a: Any, b: Any) -> bool:  # noqa: ANN401
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)
