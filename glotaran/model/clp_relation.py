""" Glotaran Relation """
from __future__ import annotations

from glotaran.model.interval_property import IntervalProperty
from glotaran.model.item import ParameterType
from glotaran.model.item import item


@item
class Relation(IntervalProperty):
    """Applies a relation between clps as

    :math:`target = parameter * source`.
    """

    source: str
    target: str
    parameter: ParameterType
