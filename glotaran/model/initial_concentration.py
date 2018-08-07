""" Glotaran Initial Concentration"""

from typing import List


class InitialConcentration(object):
    """
    An initial concentration vector.
    """
    def __init__(self, label: str, parameter: List[str]):
        self.label = label
        self.parameter = parameter

    @property
    def label(self) -> str:
        """The label of the initial concentration"""
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def parameter(self) -> List[str]:
        """A list of parameters representing the concentrations of the
        compartments."""
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        if not isinstance(value, list):
            value = [value]
        self._parameter = value

    def __str__(self):
        return f"* __{self.label}__: {self.parameter}"
