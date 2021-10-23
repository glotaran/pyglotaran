"""Glotaran package __init__.py"""
from glotaran.deprecation.modules.glotaran_root import read_model_from_yaml
from glotaran.deprecation.modules.glotaran_root import read_model_from_yaml_file
from glotaran.deprecation.modules.glotaran_root import read_parameters_from_csv_file
from glotaran.deprecation.modules.glotaran_root import read_parameters_from_yaml
from glotaran.deprecation.modules.glotaran_root import read_parameters_from_yaml_file
from glotaran.plugin_system.base_registry import load_plugins

load_plugins()

__version__ = "0.5.0rc1"


def __getattr__(attribute_name: str):
    from glotaran.deprecation.deprecation_utils import deprecate_module_attribute

    if attribute_name == "ParameterGroup":
        return deprecate_module_attribute(
            deprecated_qual_name="glotaran.ParameterGroup",
            new_qual_name="glotaran.parameter.ParameterGroup",
            to_be_removed_in_version="0.6.0",
        )

    raise AttributeError(f"module {__name__} has no attribute {attribute_name}")
