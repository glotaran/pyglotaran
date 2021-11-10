from __future__ import annotations

from pathlib import Path

from glotaran.io import load_parameters
from glotaran.io import save_parameters

DATA_PATH1 = "data/parameter.yaml"
DATA_PATH2 = "data/parameter.xlsx"
DATA_PATH3 = "testparameters.xlsx"


def test_load_parameters_xlsx():
    """load parameters file as yaml and excel and compare them"""
    parameters_xlsx = load_parameters(DATA_PATH2)
    parameters_yaml = load_parameters(DATA_PATH1)

    assert parameters_yaml == parameters_xlsx


def test_save_parameters_xlsx(
    tmp_path: Path,
):
    """load parameters file from yaml and save as xlsx
    and compare with yaml and reloaded xlsx parameter files"""

    parameters_yaml = load_parameters(DATA_PATH1)
    parameter_path = tmp_path / DATA_PATH3
    save_parameters(file_name=parameter_path, format_name="xlsx", parameters=parameters_yaml)
    parameters_saved_and_reloaded = load_parameters(parameter_path)
    parameters_prepared = load_parameters(DATA_PATH2)

    assert parameters_yaml == parameters_saved_and_reloaded
    assert parameters_prepared == parameters_saved_and_reloaded
