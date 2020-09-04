import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_files"))

LEGACY_FILES = {
    "flim_file": os.path.join(DATA_DIR, "FLIM.sdt.ascii"),
    "flim_traces": os.path.join(DATA_DIR, "FLIM_legacy_trace.ssv"),
    "flim_map": os.path.join(DATA_DIR, "FLIM_legacy_sum_map.ssv"),
}

TEMPORAL_DATA = {
    "sdt": os.path.join(DATA_DIR, "temporal.sdt"),
    "csv": os.path.join(DATA_DIR, "temporal.sdt.kin"),
}

FLIM_DATA = {
    "sdt": os.path.join(DATA_DIR, "FLIM.sdt"),
    "csv": os.path.join(DATA_DIR, "FLIM.sdt.ascii"),
}
