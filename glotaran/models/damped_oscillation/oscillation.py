"""Glotaran DOAS Model Oscillation"""


class Oscillation:
    """A damped oscillation"""
    def __init__(self, label: str, compartment: str, frequency: str, rate: str):
        """
        Parameter
        ---------
        frequency : str
            The frequency of the oscillation
        rate : str
            The damping rate of the oscillation
        """
        self._label = label
        self._compartment = compartment
        self._frequency = frequency
        self._rate = rate

    @property
    def label(self) -> str:
        """The label of the oscillation"""
        return self._label

    @property
    def compartment(self) -> str:
        """The the target compartment of the oscillation"""
        return self._compartment

    @property
    def frequency(self) -> str:
        """The frequency of the oscillation"""
        return self._frequency

    @property
    def rate(self) -> str:
        """The damping rate of the oscillation"""
        return self._rate
