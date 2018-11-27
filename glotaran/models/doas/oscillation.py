"""Glotaran DOAS Model Oscillation"""

from glotaran.model import model_item


@model_item(
    attributes={
        'frequency': str,
        'rate': str,
    })
class Oscillation:
    """A damped oscillation"""
