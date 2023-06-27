"""This module contains the interval item."""
from __future__ import annotations

from glotaran.model.item import Item
from glotaran.model.item import item


@item
class IntervalItem(Item):
    """An item with an interval."""

    interval: tuple[float, float] | list[tuple[float, float]] | None = None

    def has_interval(self) -> bool:
        """Check if intervals are defined.

        Returns
        -------
        bool
        """
        return self.interval is not None

    def applies(self, index: float | None) -> bool:
        """Check if the index is in the intervals.

        Parameters
        ----------
        index : float
            The index.

        Returns
        -------
        bool

        """
        if self.interval is None or index is None:
            return True

        def applies(interval: tuple[float, float]):
            lower, upper = interval[0], interval[1]
            if lower > upper:
                lower, upper = upper, lower
            return lower <= index <= upper

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any(applies(i) for i in self.interval)
