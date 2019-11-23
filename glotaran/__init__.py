# Glotaran package __init__.py

from . import model, parameter, io  # noqa: F401

__version__ = '0.0.15'

ParameterGroup = parameter.ParameterGroup

read_parameter_from_csv_file = ParameterGroup.from_csv
read_parameter_from_yml = ParameterGroup.from_yaml
read_parameter_from_yml_file = ParameterGroup.from_yaml_file


from .parse import parser  # noqa: E402

read_model_from_yml = parser.load_yml
read_model_from_yml_file = parser.load_yml_file

import pkg_resources  # noqa: E402

for entry_point in pkg_resources.iter_entry_points('glotaran.plugins'):
    entry_point.load()
