""" Glotaran DOAS Model Megacomplex """

from typing import List

from glotaran.model import model_item


@model_item(attributes={'oscillation': List[str]})
class DOASMegacomplex:
    """A Megacomplex with one or more oscillations."""
