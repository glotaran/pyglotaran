""" Glotaran Relation """
from __future__ import annotations

from glotaran.model.interval_property import IntervalProperty
from glotaran.model.item import model_item
from glotaran.parameter import Parameter


@model_item(
    properties={
        "source": str,
        "target": str,
        "parameter": Parameter,
    },
    has_label=False,
)
class Relation(IntervalProperty):
    """Applies a relation between clps as

    :math:`target = parameter * source`.
    """
