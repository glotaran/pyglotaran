"""This package contains compartment constraint items."""
from __future__ import annotations

from glotaran.model.interval_property import IntervalProperty
from glotaran.model.item import model_item
from glotaran.model.item import model_item_typed


@model_item(
    properties={
        "target": str,
    },
    has_type=True,
    has_label=False,
)
class OnlyConstraint(IntervalProperty):
    """A only constraint sets the calculated matrix row of a compartment to 0
    outside the given intervals."""

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


@model_item(
    properties={
        "target": str,
    },
    has_type=True,
    has_label=False,
)
class ZeroConstraint(IntervalProperty):
    """A zero constraint sets the calculated matrix row of a compartment to 0
    in the given intervals."""


@model_item_typed(
    types={
        "only": OnlyConstraint,
        "zero": ZeroConstraint,
    },
    has_label=False,
)
class Constraint:
    """A constraint is applied on one clp on one or many
    intervals on the estimated axis type.

    There are two types: zero and equal. See the documentation of
    the respective classes for details.
    """
