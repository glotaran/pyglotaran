"""Glotaran DOAS Model Oscillation"""

from glotaran.model import model_item
from glotaran.parameter import Parameter


@model_item(
    attributes={
        'frequency': Parameter,
        'rate': Parameter,
    })
class Oscillation:
    """A damped oscillation"""
