"""This package contains compartment constraint items."""
from __future__ import annotations

from glotaran.model.interval_property import IntervalProperty
from glotaran.model.item import TypedItem
from glotaran.model.item import item


@item
class Constraint(TypedItem, IntervalProperty):
    """A constraint is applied on one clp on one or many
    intervals on the estimated axis type.

    There are two types: zero and equal. See the documentation of
    the respective classes for details.
    """


@item
class ZeroConstraint(Constraint):
    """A zero constraint sets the calculated matrix row of a compartment to 0
    in the given intervals."""

    type: str = "zero"
    target: str


@item
class OnlyConstraint(ZeroConstraint):
    """A only constraint sets the calculated matrix row of a compartment to 0
    outside the given intervals."""

    type: str = "only"

    def applies(self, value: float) -> bool:
        """
        Returns true if ``value`` is in one of the intervals.

        Parameters
        ----------
        index : float

        Returns
        -------
        applies : bool

        """
        return not super().applies(value)
