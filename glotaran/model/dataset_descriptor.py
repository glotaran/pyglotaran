"""The DatasetDescriptor class."""

from typing import List

from glotaran.parameter import Parameter

from .attribute import model_attribute


@model_attribute(
    properties={
        "megacomplex": List[str],
        "scale": {"type": Parameter, "default": None, "allow_none": True},
    }
)
class DatasetDescriptor:
    """A `DatasetDescriptor` describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset describtor assigns one or more megacomplexes and a scale
    parameter.
    """
