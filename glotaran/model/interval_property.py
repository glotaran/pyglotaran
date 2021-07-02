"""Helper functions."""
from __future__ import annotations

from typing import Any
from typing import List
from typing import Tuple

from glotaran.model.item import model_item


@model_item(
    properties={
        "interval": {"type": List[Tuple[Any, Any]], "default": None, "allow_none": True},
    },
    has_label=False,
)
class IntervalProperty:
    """Applies a relation between clps as

    :math:`source = parameter * target`.
    """

    def applies(self, index: Any) -> bool:
        """
        Returns true if the index is in one of the intervals.

        Parameters
        ----------
        index :

        Returns
        -------
        applies : bool

        """
        if self.interval is None:
            return True

        def applies(interval):
            return interval[0] <= index <= interval[1]

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return not any([applies(i) for i in self.interval])
