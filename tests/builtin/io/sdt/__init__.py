from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data_files"

LEGACY_FILES = {
    "flim_file": DATA_DIR / "FLIM.sdt.ascii",
    "flim_traces": DATA_DIR / "FLIM_legacy_trace.ssv",
    "flim_map": DATA_DIR / "FLIM_legacy_sum_map.ssv",
}

TEMPORAL_DATA = {"sdt": DATA_DIR / "temporal.sdt", "csv": DATA_DIR / "temporal.sdt.kin"}

FLIM_DATA = {"sdt": DATA_DIR / "FLIM.sdt", "csv": DATA_DIR / "FLIM.sdt.ascii"}
