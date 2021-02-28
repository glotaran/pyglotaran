"""Glotaran package __init__.py"""

from . import io
from . import parameter

__version__ = "0.3.2"

ParameterGroup = parameter.ParameterGroup

read_parameters_from_csv_file = ParameterGroup.from_csv
read_parameters_from_yaml = ParameterGroup.from_yaml
read_parameters_from_yaml_file = ParameterGroup.from_yaml_file


from .parse import parser  # noqa: E402

read_model_from_yaml = parser.load_yaml
read_model_from_yaml_file = parser.load_yaml_file

import pkg_resources  # noqa: E402

for entry_point in pkg_resources.iter_entry_points("glotaran.plugins"):
    entry_point.load()
