"""Glotaran DOAS Model Oscillation"""

from glotaran.model import model_item


@model_item(
    attributes={
        'compartment': str,
        'rate': str,
        'frequency': str,
    })
class Oscillation:
    """A damped oscillation"""

    @property
    def sin_compartment(self) -> str:
        """The the sin target compartment of the oscillation"""
        return f"{self.compartment}_sin"

    @property
    def cos_compartment(self) -> str:
        """The the cos target compartment of the oscillation"""
        return f"{self.compartment}_cos"
