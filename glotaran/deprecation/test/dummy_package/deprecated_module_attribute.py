"""Dummy module to test deprecate_module_attribute."""


def __getattr__(attribute_name: str):
    from glotaran.deprecation import deprecate_module_attribute

    if attribute_name == "deprecated_attribute":
        return deprecate_module_attribute(
            deprecated_qual_name=(
                "glotaran.deprecation.test.dummy_package.deprecated_module_attribute"
            ),
            new_qual_name="glotaran.deprecation.deprecation_utils.parse_version",
            to_be_removed_in_version="0.6.0",
        )

    if attribute_name == "foo_bar":
        return deprecate_module_attribute(
            deprecated_qual_name=("glotaran.deprecation.test.dummy_package.foo_bar"),
            new_qual_name="glotaran.does_not._need_to_exists",
            to_be_removed_in_version="0.6.0",
            module_load_overwrite="glotaran.deprecation.deprecation_utils.parse_version",
        )

    raise AttributeError(f"module {__name__} has no attribute {attribute_name}")
