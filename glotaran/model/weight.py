"""This module contains weight item."""

from __future__ import annotations

from glotaran.model.item import Item


class Weight(Item):
    """The `Weight` class describes a value by which a dataset will scaled.

    `global_interval` and `model_interval` are optional. The whole range of the dataset
    will be used if not set.
    """

    global_interval: tuple[float, float] | None = None
    model_interval: tuple[float, float] | None = None
    value: float
