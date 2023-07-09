"""Glotaran package root."""
from glotaran.deprecation.deprecation_utils import deprecate_submodule
from glotaran.plugin_system.base_registry import load_plugins

load_plugins()

__version__ = "0.7.1"

examples = deprecate_submodule(
    deprecated_module_name="glotaran.examples",
    new_module_name="glotaran.testing.simulated_data",
    to_be_removed_in_version="0.8.0",
    module_load_overwrite="glotaran.deprecation.modules.examples",
)
