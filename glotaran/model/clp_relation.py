"""This module contains clp relation items."""

from __future__ import annotations

from glotaran.model.interval_item import IntervalItem
from glotaran.model.item_new import ParameterType


class ClpRelation(IntervalItem):
    """Applies a relation between two clps.

    The relation is applied as :math:`target = parameter * source`.
    """

    source: str
    target: str
    parameter: ParameterType
