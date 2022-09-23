"""This package contains compartment constraint items."""

from __future__ import annotations

from glotaran.model_new.item import Item
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import item


@item
class EqualAreaPenalty(Item):
    """An equal area constraint adds a the difference of the sum of a
    compartments in the e matrix in one or more intervals to the scaled sum
    of the e matrix of one or more target compartments to residual. The additional
    residual is scaled with the weight."""

    source: str
    source_intervals: list[tuple[float, float]]
    target: str
    target_intervals: list[tuple[float, float]]
    parameter: ParameterType
    weight: float
