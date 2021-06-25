"""This package contains compartment constraint items."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed
from glotaran.model.interval_property import IntervalProperty

if TYPE_CHECKING:
    from typing import Any


@model_attribute(
    properties={
        "target": str,
    },
    has_type=True,
    no_label=True,
)
class OnlyConstraint(IntervalProperty):
    """A only constraint sets the calculated matrix row of a compartment to 0
    outside the given intervals."""


@model_attribute(
    has_type=True,
    no_label=True,
)
class ZeroConstraint(OnlyConstraint):
    """A zero constraint sets the calculated matrix row of a compartment to 0
    in the given intervals."""

    def applies(self, index: Any) -> bool:
        """
        Returns true if the indexx is in one of the intervals.

        Parameters
        ----------
        index :

        Returns
        -------
        applies : bool

        """

        return not super().applies()


@model_attribute_typed(
    types={
        "only": OnlyConstraint,
        "zero": ZeroConstraint,
    },
    no_label=True,
)
class Constraint:
    """A constraint is applied on one clp on one or many
    intervals on the estimated axis type.

    There are two types: zero and equal. See the documentation of
    the respective classes for details.
    """

    pass
