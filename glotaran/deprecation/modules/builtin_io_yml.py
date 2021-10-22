"""Deprecation functions for the yaml parser."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.deprecation import deprecate_dict_entry

if TYPE_CHECKING:
    from typing import Any
    from typing import MutableMapping


def model_spec_deprecations(spec: MutableMapping[Any, Any]) -> None:
    """Check deprecations in the model specification ``spec`` dict.

    Parameters
    ----------
    spec : MutableMapping[Any, Any]
        Model specification dictionary
    """
    load_model_stack_level = 7
    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="type: kinetic-spectrum",
        new_usage="default_megacomplex: decay",
        to_be_removed_in_version="0.7.0",
        replace_rules=({"type": "kinetic-spectrum"}, {"default_megacomplex": "decay"}),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="type: spectral-model",
        new_usage="default_megacomplex: spectral",
        to_be_removed_in_version="0.7.0",
        replace_rules=({"type": "spectral-model"}, {"default_megacomplex": "spectral"}),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="default-megacomplex",
        new_usage="default_megacomplex",
        to_be_removed_in_version="0.7.0",
        swap_keys=("default-megacomplex", "default_megacomplex"),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="spectral_relations",
        new_usage="clp_relations",
        to_be_removed_in_version="0.7.0",
        swap_keys=("spectral_relations", "clp_relations"),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="relations",
        new_usage="clp_relations",
        to_be_removed_in_version="0.7.0",
        swap_keys=("relations", "clp_relations"),
        stacklevel=load_model_stack_level,
    )

    if "clp_relations" in spec:
        for relation in spec["clp_relations"]:
            deprecate_dict_entry(
                dict_to_check=relation,
                deprecated_usage="clp_relations:\n - compartment",
                new_usage="clp_relations:\n - source",
                to_be_removed_in_version="0.7.0",
                swap_keys=("compartment", "source"),
                stacklevel=load_model_stack_level,
            )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="spectral_constraints",
        new_usage="clp_constraints",
        to_be_removed_in_version="0.7.0",
        swap_keys=("spectral_constraints", "clp_constraints"),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="constraints",
        new_usage="clp_constraints",
        to_be_removed_in_version="0.7.0",
        swap_keys=("constraints", "clp_constraints"),
        stacklevel=load_model_stack_level,
    )

    if "clp_constraints" in spec:
        for constraint in spec["clp_constraints"]:
            deprecate_dict_entry(
                dict_to_check=constraint,
                deprecated_usage="clp_constraints:\n - compartment",
                new_usage="clp_constraints:\n - target",
                to_be_removed_in_version="0.7.0",
                swap_keys=("compartment", "target"),
                stacklevel=load_model_stack_level,
            )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="equal_area_penalties",
        new_usage="clp_area_penalties",
        to_be_removed_in_version="0.7.0",
        swap_keys=("equal_area_penalties", "clp_area_penalties"),
        stacklevel=load_model_stack_level,
    )

    if "irf" in spec:
        for _, irf in spec["irf"].items():
            deprecate_dict_entry(
                dict_to_check=irf,
                deprecated_usage="center_dispersion",
                new_usage="center_dispersion_coefficients",
                to_be_removed_in_version="0.7.0",
                swap_keys=("center_dispersion", "center_dispersion_coefficients"),
                stacklevel=load_model_stack_level,
            )

        for _, irf in spec["irf"].items():
            deprecate_dict_entry(
                dict_to_check=irf,
                deprecated_usage="width_dispersion",
                new_usage="width_dispersion_coefficients",
                to_be_removed_in_version="0.7.0",
                swap_keys=("width_dispersion", "width_dispersion_coefficients"),
                stacklevel=load_model_stack_level,
            )


def scheme_spec_deprecations(spec: MutableMapping[Any, Any]) -> None:
    """Check deprecations in the scheme specification ``spec`` dict.

    Parameters
    ----------
    spec : MutableMapping[Any, Any]
        Scheme specification dictionary
    """
    load_scheme_stack_level = 7

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="maximum-number-function-evaluations: <number>",
        new_usage="maximum_number_function_evaluations: <number>",
        to_be_removed_in_version="0.7.0",
        swap_keys=("maximum-number-function-evaluations", "maximum_number_function_evaluations"),
        stacklevel=load_scheme_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="non-negative-least-squares",
        new_usage=("<model_file>dataset_groups.default.residual_function"),
        to_be_removed_in_version="0.7.0",
        swap_keys=("non-negative-least-squares", "non_negative_least_squares"),
        stacklevel=load_scheme_stack_level,
    )
