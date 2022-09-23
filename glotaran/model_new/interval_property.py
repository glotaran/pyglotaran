from __future__ import annotations

from glotaran.model_new.item import Item
from glotaran.model_new.item import item


@item
class IntervalProperty(Item):
    interval: tuple[float, float] | list[tuple[float, float]] | None = None

    def has_interval(self) -> bool:
        return self.interval is not None

    def applies(self, value: float | None) -> bool:
        """
        Returns true if ``value`` is in one of the intervals.

        Parameters
        ----------
        value : float

        Returns
        -------
        applies : bool

        """
        if self.interval is None or value is None:
            return True

        def applies(interval: tuple[float, float]):
            lower, upper = interval[0], interval[1]
            if lower > upper:
                lower, upper = upper, lower
            return lower <= value <= upper

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any(applies(i) for i in self.interval)
