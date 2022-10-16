"""This module contains clp constraint items."""
from __future__ import annotations

from glotaran.model.interval_item import IntervalItem
from glotaran.model.item import TypedItem
from glotaran.model.item import item


@item
class ClpConstraint(TypedItem, IntervalItem):
    """Baseclass for clp constraints.

    There are two types: zero and equal. See the documentation of
    the respective classes for details.
    """

    target: str


@item
class ZeroConstraint(ClpConstraint):
    """Constraints the target to 0 in the given interval."""

    type: str = "zero"


@item
class OnlyConstraint(ZeroConstraint):
    """Constraints the target to 0 outside the given interval."""

    type: str = "only"

    def applies(self, index: float | None) -> bool:
        """Check if the constraint applies on this index.

        Parameters
        ----------
        index : float
            The index.

        Returns
        -------
        bool
        """
        return not super().applies(index)
