"""This module contains clp penalty items."""

from __future__ import annotations

from typing import Literal

from glotaran.model.item import Attribute
from glotaran.model.item import Item
from glotaran.model.item import ParameterType


class ClpPenalty(Item):
    """Baseclass for clp penalties."""


class EqualAreaPenalty(ClpPenalty):
    """Penalizes a difference between the area of one amplitude and the scaled area of another.

    The area is the sum of the amplitude over its intervals on the global dimension.
    The term ``weight * |area(source) - parameter * area(target)|`` is added to the residual.
    """

    # note: we do not use pydantic discriminators as it needs at least 2 different types.
    # we keep the type though for later extension.
    type: Literal["equal_area"]
    source: str = Attribute(description="Label of the amplitude whose area is compared.")
    source_intervals: list[tuple[float, float]] | tuple[float, float] | None = Attribute(
        default=None,
        description=(
            "Intervals on the global dimension over which the source is summed. "
            "Without it, the whole range."
        ),
    )
    target: str = Attribute(description="Label of the amplitude whose area is scaled.")
    target_intervals: list[tuple[float, float]] | tuple[float, float] | None = Attribute(
        default=None,
        description=(
            "Intervals on the global dimension over which the target is summed. "
            "Without it, the whole range."
        ),
    )
    parameter: ParameterType = Attribute(
        description="The factor of the target area: the penalty is "
        "weight * |area(source) - parameter * area(target)|."
    )
    weight: float = Attribute(description="The factor of the penalty in the residual.")

    def source_applies(self, index: float | None) -> bool:
        return self.applies(index, self.source_intervals)  # type:ignore[arg-type]

    def target_applies(self, index: float | None) -> bool:
        return self.applies(index, self.target_intervals)  # type:ignore[arg-type]

    def applies(self, index: float | None, intervals: list[tuple[float, float]] | None) -> bool:
        """Check if the index is in the intervals.

        Parameters
        ----------
        index : float
            The index.

        Returns
        -------
        bool

        """
        if intervals is None or index is None:
            return True

        def applies(interval: tuple[float, float]) -> bool:
            lower, upper = interval[0], interval[1]
            if lower > upper:
                lower, upper = upper, lower
            return lower <= index <= upper

        if isinstance(intervals, tuple):
            return applies(intervals)
        return any(applies(i) for i in intervals)
