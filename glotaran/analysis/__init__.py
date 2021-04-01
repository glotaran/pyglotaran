"""This package contains functions for model simulation and fitting."""
from glotaran.deprecation.deprecation_utils import deprecate_submodule

result = deprecate_submodule(
    deprecated_module_name="glotaran.analysis.result",
    new_module_name="glotaran.project.result",
    to_be_removed_in_version="0.6.0",
)

scheme = deprecate_submodule(
    deprecated_module_name="glotaran.analysis.scheme",
    new_module_name="glotaran.project.scheme",
    to_be_removed_in_version="0.6.0",
)
