from __future__ import annotations

from pathlib import Path

# from glotaran.examples.sequential import model
from glotaran.io import load_parameters
from glotaran.io import save_parameters

DATA_PATH1 = "data/parameter.yaml"  # stored in /data
DATA_PATH2 = "testparameters.xlsx"  # saved during test
DATA_PATH3 = "data/parameter.xlsx"  # stored in /data


def test_save_parameters(
    tmp_path: Path,
):
    parameters_yaml = load_parameters(DATA_PATH1)

    parameter_path = tmp_path / DATA_PATH2
    save_parameters(file_name=parameter_path, format_name="xlsx", parameters=parameters_yaml)
    parameters_saved_and_reloaded = load_parameters(parameter_path)

    parameters_prepared = load_parameters(DATA_PATH3)

    assert parameters_prepared == parameters_saved_and_reloaded
