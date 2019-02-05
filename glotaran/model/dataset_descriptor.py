"""This package contains glotarans dataset descriptor."""

from typing import List

from .model_attribute import model_attribute
from glotaran.parameter import Parameter


@model_attribute(properties={
    'megacomplex': List[str],
    'scale': {'type': Parameter, 'default': None, 'allow_none': True},
})
class DatasetDescriptor:
    """A dataset descriptor describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset describtor assigns one or more megacomplexes, a scale
    parameter and a set of compartment constrains to a dataset.
    """
