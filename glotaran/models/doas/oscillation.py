"""Glotaran DOAS Model Oscillation"""

from glotaran.model import model_attribute
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        'frequency': Parameter,
        'rate': Parameter,
    })
class Oscillation:
    """A damped oscillation"""
