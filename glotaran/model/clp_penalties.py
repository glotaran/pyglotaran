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
    source_intervals: list[tuple[float, float]]
    target: str
    target_intervals: list[tuple[float, float]]
    parameter: ParameterType
    weight: float
