""" Gloataran DOAS Model """

from typing import Dict

from glotaran.models.spectral_temporal import KineticModel

from .oscillation import Oscillation


class DOASModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
    def __init__(self):
        """ """
        self.oscillations = {}
        super(DOASModel, self).__init__()

    @property
    def oscillations(self) -> Dict[str, Oscillation]:
        """A dictonary of the models oscillations."""
        return self._oscillations

    @oscillations.setter
    def oscillations(self, value: Dict[str, Oscillation]):
        if not isinstance(value, dict):
            raise TypeError("Oscillations must be dict.")
        if any(not isinstance(type(val), Oscillation) for val in value):
            raise TypeError("Oscillations must be subclass of 'Oscillation'")
        self._oscillations = value

    def add_oscillation(self, oscillation):
        """

        Parameters
        ----------


        Returns
        -------

        """
        if not issubclass(type(oscillation), Oscillation):
            raise TypeError("Oscillation must be subclass of 'Oscillation")
        if self.oscillations is None:
            self.oscillations = {oscillation.label: oscillation}
        else:
            if oscillation.label in self.oscillations:
                raise Exception("Oscillation labels must be unique")
            self.oscillations[oscillation.label] = oscillation
