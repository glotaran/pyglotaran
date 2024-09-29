"""This module contains clp constraint items."""

from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import AfterValidator  # noqa: TCH002

from glotaran.model.interval_item import IntervalItem
from glotaran.model.item import TypedItem


class ClpConstraint(TypedItem, IntervalItem):
    """Baseclass for clp constraints.

    There are two types: zero and equal. See the documentation of
    the respective classes for details.
    """

    target: Annotated[
        str | list[str], AfterValidator(lambda v: [v] if isinstance(v, str) else v)
    ]


class ZeroConstraint(ClpConstraint):
    """Constraints the target to 0 in the given interval."""

    type: Literal["zero"]  # type:ignore[assignment]


class OnlyConstraint(ZeroConstraint):
    """Constraints the target to 0 outside the given interval."""

    type: Literal["only"]  # type:ignore[assignment]

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
