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
    def sin_compartment(self) -> str:
        """The the sin target compartment of the oscillation"""
        return f"{self._compartment}_sin"

    @property
    def cos_compartment(self) -> str:
        """The the cos target compartment of the oscillation"""
        return f"{self._compartment}_cos"

    @property
    def frequency(self) -> str:
        """The frequency of the oscillation"""
        return self._frequency

    @property
    def rate(self) -> str:
        """The damping rate of the oscillation"""
        return self._rate

    def __str__(self):
        string = f"_{self.label}_:"
        string += f" __Compartment__: {self.compartment}"
        string += f" __Rate__: {self.rate}"
        string += f" __Frequency__: {self.frequency}"
        return string
