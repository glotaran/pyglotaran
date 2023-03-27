"""Deprecation functions for the yaml parser."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.deprecation import deprecate_dict_entry

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any


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
        deprecated_usage="clp_area_penalties",
        new_usage="clp_penalties",
        to_be_removed_in_version="0.8.0",
        swap_keys=("clp_area_penalties", "clp_penalties"),
        stacklevel=load_model_stack_level,
    )
