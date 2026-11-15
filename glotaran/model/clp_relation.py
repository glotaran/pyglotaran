"""This module contains clp relation items."""

from __future__ import annotations

from glotaran.model.interval_item import IntervalItem
from glotaran.model.item import Attribute
from glotaran.model.item import ParameterType


class ClpRelation(IntervalItem):
    """Sets one amplitude to a factor times another: ``target = parameter * source``."""

    source: str = Attribute(description="Label of the amplitude the target is computed from.")
    target: str = Attribute(description="Label of the amplitude that is replaced by the relation.")
    parameter: ParameterType = Attribute(
        description="The factor of the relation: target = parameter * source."
    )
