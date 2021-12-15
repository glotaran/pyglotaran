from __future__ import annotations

from pathlib import Path

from glotaran.io import load_parameters
from glotaran.io import save_parameters

PANDAS_TEST_DATA = Path(__file__).parent / "data"
PATH_XLSX = PANDAS_TEST_DATA / "parameter.xlsx"
PATH_ODS = PANDAS_TEST_DATA / "parameter.ods"
PATH_CSV = PANDAS_TEST_DATA / "parameter.csv"
PATH_TSV = PANDAS_TEST_DATA / "parameter.tsv"
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


def test_load_parameters_ods():
    """load parameters file as yaml and ods and compare them"""
    parameters_ods = load_parameters(PATH_ODS)
    parameters_yaml = load_parameters(PATH_YAML)
    assert parameters_yaml == parameters_ods


def test_save_parameters_ods(
    tmp_path: Path,
):
    """load parameters file from yaml and save as ods
    and compare with yaml and reloaded ods parameter files"""
    parameters_yaml = load_parameters(PATH_YAML)
    parameter_path = tmp_path / "testparameters.ods"
    save_parameters(file_name=parameter_path, format_name="ods", parameters=parameters_yaml)
    parameters_saved_and_reloaded = load_parameters(parameter_path)
    parameters_ods = load_parameters(PATH_ODS)
    assert parameters_yaml == parameters_saved_and_reloaded
    assert parameters_ods == parameters_saved_and_reloaded


def test_load_parameters_csv():
    """load parameters file as yaml and csv and compare them"""
    parameters_csv = load_parameters(PATH_CSV)
    parameters_yaml = load_parameters(PATH_YAML)
    assert parameters_yaml == parameters_csv


def test_save_parameters_csv(
    tmp_path: Path,
):
    """load parameters file from yaml and save as csv
    and compare with yaml and reloaded csv parameter files"""
    parameters_yaml = load_parameters(PATH_YAML)
    parameter_path = tmp_path / "testparameters.csv"
    save_parameters(file_name=parameter_path, format_name="csv", parameters=parameters_yaml)
    parameters_saved_and_reloaded = load_parameters(parameter_path)
    parameters_csv = load_parameters(PATH_CSV)
    assert parameters_yaml == parameters_saved_and_reloaded
    assert parameters_csv == parameters_saved_and_reloaded


def test_load_parameters_tsv():
    """load parameters file as yaml and tsv and compare them"""
    parameters_tsv = load_parameters(PATH_TSV)
    parameters_yaml = load_parameters(PATH_YAML)
    assert parameters_yaml == parameters_tsv


def test_save_parameters_tsv(
    tmp_path: Path,
):
    """load parameters file from yaml and save as tsv
    and compare with yaml and reloaded tsv parameter files"""
    parameters_yaml = load_parameters(PATH_YAML)
    parameter_path = tmp_path / "testparameters.tsv"
    save_parameters(file_name=parameter_path, format_name="tsv", parameters=parameters_yaml)
    parameters_saved_and_reloaded = load_parameters(parameter_path)
    parameters_tsv = load_parameters(PATH_TSV)
    assert parameters_yaml == parameters_saved_and_reloaded
    assert parameters_tsv == parameters_saved_and_reloaded
