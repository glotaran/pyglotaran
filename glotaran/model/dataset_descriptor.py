"""This package contains glotarans dataset descriptor."""

from typing import List

from .model_item import model_item


@model_item(attributes={
    'megacomplex': List[str],
    'scale': {'type': str, 'default': None},
})
class DatasetDescriptor:
    """A dataset descriptor describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset describtor assigns one or more megacomplexes, a scale
    parameter and a set of compartment constrains to a dataset.
    """
