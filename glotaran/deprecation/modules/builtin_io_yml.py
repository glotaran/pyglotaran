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
        new_usage="default-megacomplex: decay",
        to_be_removed_in_version="0.7.0",
        replace_rules=({"type": "kinetic-spectrum"}, {"default-megacomplex": "decay"}),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="type: spectral-model",
        new_usage="default-megacomplex: spectral",
        to_be_removed_in_version="0.7.0",
        replace_rules=({"type": "spectral-model"}, {"default-megacomplex": "spectral"}),
        stacklevel=load_model_stack_level,
    )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="spectral_relations",
        new_usage="relations",
        to_be_removed_in_version="0.7.0",
        swap_keys=("spectral_relations", "relations"),
        stacklevel=load_model_stack_level,
    )

    if "relations" in spec:
        for relation in spec["relations"]:
            deprecate_dict_entry(
                dict_to_check=relation,
                deprecated_usage="compartment",
                new_usage="source",
                to_be_removed_in_version="0.7.0",
                swap_keys=("compartment", "source"),
                stacklevel=load_model_stack_level,
            )

    deprecate_dict_entry(
        dict_to_check=spec,
        deprecated_usage="spectral_constraints",
        new_usage="constraints",
        to_be_removed_in_version="0.7.0",
        swap_keys=("spectral_constraints", "constraints"),
        stacklevel=load_model_stack_level,
    )

    if "constraints" in spec:
        for constraint in spec["constraints"]:
            deprecate_dict_entry(
                dict_to_check=constraint,
                deprecated_usage="constraint.compartment",
                new_usage="constraint.target",
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
