"""This package contains functions for model simulation and fitting."""
from glotaran.deprecation.deprecation_utils import deprecate_submodule

simulation = deprecate_submodule(
    deprecated_module_name="glotaran.analysis.simulation",
    new_module_name="glotaran.simulation.simulation",
    to_be_removed_in_version="0.8.0",
)

optimize = deprecate_submodule(
    deprecated_module_name="glotaran.analysis.optimize",
    new_module_name="glotaran.optimization.optimize",
    to_be_removed_in_version="0.8.0",
)
