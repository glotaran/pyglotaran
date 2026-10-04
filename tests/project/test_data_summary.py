from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.io import load_dataset
from glotaran.io import save_dataset
from glotaran.project.data_summary import compare_data_summaries
from glotaran.project.data_summary import data_source_path
from glotaran.project.data_summary import summarize_data

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"data": (("time", "spectral"), rng.uniform(1, 2, size=(50, 4)))},
        coords={"time": np.linspace(-1, 10, 50), "spectral": [400.0, 500.0, 600.0, 700.0]},
    )


def test_summarize_data(dataset: xr.Dataset):
    summary = summarize_data(dataset)
    assert summary["shape"] == {"time": 50, "spectral": 4}
    assert summary["time"] == pytest.approx(
        {"min": -1, "max": 10, "mean": 4.5, "rms": np.sqrt(np.mean(dataset.time.values**2))}
    )
    assert summary["spectral"]["mean"] == 550
    assert summary["data"]["rms"] == pytest.approx(np.sqrt(np.mean(dataset.data.values**2)))
    assert "weight" not in summary


def test_identical_transposed_and_tiny_differences_are_equal(dataset: xr.Dataset):
    summary = summarize_data(dataset)
    assert compare_data_summaries("ta", summary, summarize_data(dataset.copy(deep=True))) == []

    transposed = dataset.transpose("spectral", "time")
    transposed["data"] = transposed.data.astype(np.float32).astype(np.float64)
    assert summarize_data(dataset.transpose("spectral", "time")) == summary
    assert compare_data_summaries("ta", summary, summarize_data(transposed)) == []

    shifted = dataset.copy(deep=True)
    shifted["data"] = shifted.data + 1e-9
    assert compare_data_summaries("ta", summary, summarize_data(shifted)) == []


def test_changes_are_reported(dataset: xr.Dataset):
    summary = summarize_data(dataset)

    differences = compare_data_summaries(
        "ta", summary, summarize_data(dataset.isel(time=slice(0, 40)))
    )
    assert differences[0] == "ta: shape {'time': 50, 'spectral': 4} → {'time': 40, 'spectral': 4}"
    assert any(difference.startswith("ta: time max 10 → ") for difference in differences)

    scaled = dataset.copy(deep=True)
    scaled["data"] = scaled.data * 1.004
    assert compare_data_summaries("ta", summary, summarize_data(scaled)) == [
        f"ta: data {statistic} {summary['data'][statistic]:.6g} → "
        f"{summary['data'][statistic] * 1.004:.6g} (+0.40%)"
        for statistic in ("min", "max", "mean", "rms")
    ]

    weighted = dataset.copy()
    weighted["weight"] = xr.ones_like(dataset.data)
    assert compare_data_summaries("ta", summary, summarize_data(weighted)) == ["ta: weight added"]
    assert compare_data_summaries("ta", summarize_data(weighted), summary) == [
        "ta: weight removed"
    ]


def test_data_source_path(dataset: xr.Dataset, tmp_path: Path):
    record_folder = tmp_path / "results" / "2026-10-04_14-28-05"
    assert data_source_path(dataset, record_folder) is None

    save_dataset(dataset, tmp_path / "data" / "ta.nc")
    loaded = load_dataset(tmp_path / "data" / "ta.nc")
    assert data_source_path(loaded, record_folder) == "../../data/ta.nc"
    # Preprocessing keeps the attributes, so the reference names the original file
    preprocessed = loaded.isel(time=slice(0, 10)) * 1000
    preprocessed.attrs = loaded.attrs
    assert data_source_path(preprocessed, record_folder) == "../../data/ta.nc"
