"""Module containing a deprecated fake module."""
from glotaran.deprecation.deprecation_utils import deprecate_submodule

# just here to be tested
deprecated_module = deprecate_submodule(
    deprecated_module_name="glotaran.deprecation.test.dummy_package.deprecated_module",
    new_module_name="glotaran.deprecation.deprecation_utils",
    to_be_removed_in_version="0.6.0",
)
