""" Glotaran DOAS Model Megacomplex """

from typing import List

from glotaran.models.spectral_temporal import KineticMegacomplex


class DOASMegacomplex(KineticMegacomplex):
    """Extends the glotaran.models.spectral_temporal.Megacomplex with
    oscillations"""
    def __init__(self, label, k_matrices: List[str], oscillations: List[str]):
        if not isinstance(oscillations, list):
            oscillations = [oscillations]
        self._oscillations = oscillations
        super(DOASMegacomplex, self).__init__(label, k_matrices)

    @property
    def oscillations(self):
        return self._oscillations

    @oscillations.setter
    def oscillations(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(m, str) for m in value):
            raise TypeError
        self._oscillations = value

    def __str__(self):
        string = super(DOASMegacomplex, self).__str__()
        string += f"* _Oscillations_: {self.oscillations}\n"
        return string
