""" Glotaran DOAS Model Megacomplex """

from typing import List

from glotaran.model import model_attribute
from glotaran.models.spectral_temporal import KineticMegacomplex


@model_attribute(properties={'oscillation': {'type': List[str], 'default': []}})
class DOASMegacomplex(KineticMegacomplex):
    """A Megacomplex with one or more oscillations."""
