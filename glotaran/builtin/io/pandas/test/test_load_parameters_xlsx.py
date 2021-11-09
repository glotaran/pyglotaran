# not working...

from __future__ import annotations

# from glotaran.examples.sequential import model
from glotaran.io import load_parameters

DATA_PATH1 = "data/parameter.yaml"
DATA_PATH2 = "data/parameter.xlsx"


# loading parameters file as yaml and excel and compare them
def test_load_parameters():
    # DATA_DIR = Path(__file__).parent
    # TEST_FILE_YML = DATA_DIR.joinpath("data/parameter.yaml")
    # data_file = ExplicitFile(TEST_FILE_YML)
    # load_parameters(parameters=parameter, format_name="yml", file_name="parameter")
    parameters_xlsx = load_parameters(DATA_PATH2)
    parameters_yaml = load_parameters(DATA_PATH1)
    # parameters_yaml = load_parameters("parameter_mock.yaml")
    assert parameters_yaml == parameters_xlsx
