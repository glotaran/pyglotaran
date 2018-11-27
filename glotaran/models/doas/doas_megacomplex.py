""" Glotaran DOAS Model Megacomplex """

from typing import List

from glotaran.model import model_item
from glotaran.models.spectral_temporal import KineticMegacomplex


@model_item(attributes={'oscillation': {'type': List[str], 'default': []}})
class DOASMegacomplex(KineticMegacomplex):
    """A Megacomplex with one or more oscillations."""
