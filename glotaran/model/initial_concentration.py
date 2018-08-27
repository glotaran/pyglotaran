""" Glotaran Initial Concentration"""

from typing import List
from .decorators import glotaran_model_item


@glotaran_model_item(attributes={'parameters': List[str]},
                     parameter=['parameters'])
class InitialConcentration(object):

    def __str__(self):
        return f"* __{self.label}__: {self.parameters}"
