"""Glotaran package __init__.py"""

from . import parameter

__version__ = "0.3.0"

ParameterGroup = parameter.ParameterGroup

read_parameters_from_csv_file = ParameterGroup.from_csv
read_parameters_from_yaml = ParameterGroup.from_yaml
read_parameters_from_yaml_file = ParameterGroup.from_yaml_file


#  from .parse import parser
#
#  read_model_from_yaml = parser.load_yaml
#  read_model_from_yaml_file = parser.load_yaml_file

from glotaran.register import register  # noqa: E402


def load_model(file_name: str, fmt=None):
    if fmt is None:
        fmt = file_name.split(".")
        if len(fmt) == 1:
            raise ValueError(f"Cannot determine format of modelfile '{file_name}'")
        fmt = fmt[-1]
    io = register.get_io(fmt)

    try:
        return io.read_model(fmt, file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read models with format '{fmt}'")


import pkg_resources  # noqa: E402

for entry_point in pkg_resources.iter_entry_points("glotaran.plugins"):
    entry_point.load()
