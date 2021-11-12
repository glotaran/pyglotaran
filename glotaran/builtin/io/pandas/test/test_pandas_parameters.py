from __future__ import annotations

from pathlib import Path

from glotaran.io import load_parameters
from glotaran.io import save_parameters

PANDAS_TEST_DATA = Path(__file__).parent / "data"
PATH_XLSX = PANDAS_TEST_DATA / "parameter.xlsx"
PATH_YAML = PANDAS_TEST_DATA / "parameter.yaml"


def test_load_parameters_xlsx():
    """load parameters file as yaml and excel and compare them"""
    parameters_xlsx = load_parameters(PATH_XLSX)
    parameters_yaml = load_parameters(PATH_YAML)
    assert parameters_yaml == parameters_xlsx


def test_save_parameters_xlsx(
    tmp_path: Path,
):
    """load parameters file from yaml and save as xlsx
    and compare with yaml and reloaded xlsx parameter files"""
    parameters_yaml = load_parameters(PATH_YAML)
    parameter_path = tmp_path / "testparameters.xlsx"
    save_parameters(file_name=parameter_path, format_name="xlsx", parameters=parameters_yaml)
    parameters_saved_and_reloaded = load_parameters(parameter_path)
    parameters_xlsx = load_parameters(PATH_XLSX)
    assert parameters_yaml == parameters_saved_and_reloaded
    assert parameters_xlsx == parameters_saved_and_reloaded
