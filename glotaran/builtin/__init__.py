"""This package contains builtin plugins."""
from glotaran.deprecation.deprecation_utils import deprecate_submodule

read_data_file = deprecate_submodule(
    deprecated_module_name="glotaran.builtin.read_data_file",
    new_module_name="glotaran.builtin.io",
    to_be_removed_in_version="0.6.0",
)
