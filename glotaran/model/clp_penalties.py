"""This module contains clp penalty items."""

from __future__ import annotations

from typing import Literal

from glotaran.model.item import Item
from glotaran.model.item import ParameterType


class ClpPenalty(Item):
    """Baseclass for clp penalties."""


class EqualAreaPenalty(ClpPenalty):
    """Forces the area of 2 clp to be the same.

    An equal area constraint adds a the difference of the sum of a
    compartments in the e matrix in one or more intervals to the scaled sum
    of the e matrix of one or more target compartments to residual. The additional
    residual is scaled with the weight.
    """

    # note: we do not use pydantic discriminators as it needs at least 2 different types.
    # we keep the type though for later extension.
    type: Literal["equal_area"]
    source: str
    source_intervals: list[tuple[float, float]] | tuple[float, float] | None = None
    target: str
    target_intervals: list[tuple[float, float]] | tuple[float, float] | None = None
    parameter: ParameterType
    weight: float

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
