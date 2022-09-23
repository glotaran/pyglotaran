""" Glotaran Relation """
from __future__ import annotations

from glotaran.model_new.interval_property import IntervalProperty
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import item


@item
class Relation(IntervalProperty):
    """Applies a relation between clps as

    :math:`target = parameter * source`.
    """

    source: str
    target: str
    parameter: ParameterType
