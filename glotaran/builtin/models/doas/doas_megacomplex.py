""" Glotaran DOAS Model Megacomplex """

from typing import List

from glotaran.model import model_attribute
from glotaran.builtin.models.kinetic_image.kinetic_image_megacomplex \
    import KineticImageMegacomplex


@model_attribute(properties={'oscillation': {'type': List[str], 'default': []}})
class DOASMegacomplex(KineticImageMegacomplex):
    """A Megacomplex with one or more oscillations."""
