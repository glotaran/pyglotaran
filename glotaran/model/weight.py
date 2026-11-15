"""This module contains weight item."""

from __future__ import annotations

from glotaran.model.item import Attribute
from glotaran.model.item import Item


class Weight(Item):
    """A weight for a region of a dataset.

    Without ``global_interval`` and ``model_interval`` the weight applies to the whole dataset.
    """

    global_interval: tuple[float, float] | None = Attribute(
        default=None,
        description=(
            "Interval `[minimum, maximum]` on the global dimension where the weight applies. "
            "Without it, the whole range."
        ),
    )
    model_interval: tuple[float, float] | None = Attribute(
        default=None,
        description=(
            "Interval `[minimum, maximum]` on the model dimension where the weight applies. "
            "Without it, the whole range."
        ),
    )
    value: float = Attribute(description="The weight of the data in the intervals.")
