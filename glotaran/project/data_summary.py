"""Summaries of input data, used to recognize the data of a recorded fit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from glotaran.utils.io import relative_posix_path

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

STATISTICS = ("min", "max", "mean", "rms")
SUMMARY_RTOL = 1e-6
"""Differences up to this factor times the RMS of an array count as equal."""


def summarize_values(values: np.ndarray) -> dict[str, float]:
    """Summarize an array by its minimum, maximum, mean and root mean square in float64.

    Parameters
    ----------
    values : np.ndarray
        Values to summarize, in a fixed (sorted) dimension order and C order, so that the
        summation order does not depend on the layout of the original array.

    Returns
    -------
    dict[str, float]
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def summarize_data(dataset: xr.Dataset) -> dict[str, Any]:
    """Summarize a dataset as passed to ``optimize``.

    The summary holds the shape by dimension name and the statistics of each coordinate of the
    dimensions of ``data`` and of the variables ``data`` and, if present, ``weight``.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with a ``data`` variable.

    Returns
    -------
    dict[str, Any]
    """
    data = dataset.data
    summary: dict[str, Any] = {"shape": {str(dim): int(size) for dim, size in data.sizes.items()}}
    for dim in data.dims:
        if dim in dataset.coords and np.issubdtype(dataset.coords[dim].dtype, np.number):
            summary[str(dim)] = summarize_values(dataset.coords[dim].to_numpy())
    for name in ("data", "weight"):
        if name in dataset:
            summary[name] = summarize_values(
                dataset[name].transpose(*sorted(dataset[name].dims)).to_numpy()
            )
    return summary


def data_source_path(dataset: xr.Dataset, relative_to: Path) -> str | None:
    """Return the file the dataset was loaded from, relative to ``relative_to`` where possible.

    This is an unchecked reference: preprocessing after loading keeps the attribute.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset.
    relative_to : Path
        Folder the path is made relative to.

    Returns
    -------
    str | None
        ``None`` for data not loaded with ``load_dataset``, the only function that sets
        ``io_plugin_name`` next to ``source_path``.
    """
    if "io_plugin_name" not in dataset.attrs or "source_path" not in dataset.attrs:
        return None
    return relative_posix_path(dataset.attrs["source_path"], base_path=relative_to)


def compare_data_summaries(
    label: str, recorded: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Describe what changed between two data summaries and by how much.

    ``source_path`` is not compared. Statistics differing by up to ``SUMMARY_RTOL`` times the
    RMS of the array count as equal.

    Parameters
    ----------
    label : str
        Dataset label used in the messages.
    recorded : dict[str, Any]
        Summary of the recorded fit.
    current : dict[str, Any]
        Summary of the current data.

    Returns
    -------
    list[str]
        One message per difference, for example ``"ta: time max 10 -> 8 (-20 %)"``; empty if the
        summaries are equal.
    """
    differences = []
    if recorded.get("shape") != current.get("shape"):
        differences.append(f"{label}: shape {recorded.get('shape')} -> {current.get('shape')}")
    names = [name for name in recorded if isinstance(recorded[name], dict) and name != "shape"]
    names += [name for name in current if isinstance(current[name], dict) and name != "shape"]
    for name in dict.fromkeys(names):
        if name not in current:
            differences.append(f"{label}: {name} removed")
            continue
        if name not in recorded:
            differences.append(f"{label}: {name} added")
            continue
        tolerance = SUMMARY_RTOL * max(
            abs(recorded[name].get("rms", 0)), abs(current[name].get("rms", 0))
        )
        for statistic in STATISTICS:
            old, new = recorded[name].get(statistic), current[name].get(statistic)
            if old is None or new is None or _equal(old, new, tolerance):
                continue
            change = f" ({(new - old) / abs(old):+.2%})" if old != 0 else ""
            differences.append(f"{label}: {name} {statistic} {old:.6g} -> {new:.6g}{change}")
    return differences


def _equal(old: float, new: float, tolerance: float) -> bool:
    if math.isnan(old) or math.isnan(new):
        return math.isnan(old) and math.isnan(new)
    return old == new or abs(new - old) <= tolerance
