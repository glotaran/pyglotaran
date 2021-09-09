"""Helper functions."""
from __future__ import annotations

from typing import List
from typing import Tuple

from glotaran.model.item import model_item


@model_item(
    properties={
        "interval": {"type": List[Tuple[float, float]], "default": None, "allow_none": True},
    },
    has_label=False,
)
class IntervalProperty:
    """Applies a relation between clps as

    :math:`source = parameter * target`.
    """

    def applies(self, value: float) -> bool:
        """
        Returns true if ``value`` is in one of the intervals.

        Parameters
        ----------
        value : float

        Returns
        -------
        applies : bool

        """
        if self.interval is None:
            return True

        def applies(interval):
            return interval[0] <= value <= interval[1]

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any([applies(i) for i in self.interval])
