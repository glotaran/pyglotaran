""" Glotaran Initial Concentration"""

from typing import List
from .model_item import glotaran_model_item


@glotaran_model_item(attributes={'parameters': List[str]})
class InitialConcentration:
    pass
