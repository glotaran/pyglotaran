"""This module contains clp relation items."""

from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.model.interval_item import IntervalItem

if TYPE_CHECKING:
    from glotaran.model.item import ParameterType


class ClpRelation(IntervalItem):
    """Applies a relation between two clps.

    The relation is applied as :math:`target = parameter * source`.
    """

    source: str
    target: str
    parameter: ParameterType
