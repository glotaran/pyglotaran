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

    raise AttributeError(f"module {__name__} has no attribute {attribute_name}")
