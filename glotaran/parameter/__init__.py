"""The glotaran parameter package."""
from glotaran.parameter.parameter import Parameter
from glotaran.parameter.parameter_history import ParameterHistory
from glotaran.parameter.parameters import Parameters


def __getattr__(attribute_name: str):
    from glotaran.deprecation import deprecate_module_attribute

    if attribute_name == "ParameterGroup":
        return deprecate_module_attribute(
            deprecated_qual_name="glotaran.parameter.ParameterGroup",
            new_qual_name="glotaran.parameter.Parameters",
            to_be_removed_in_version="0.8.0",
        )

    raise AttributeError(f"module {__name__} has no attribute {attribute_name}")
