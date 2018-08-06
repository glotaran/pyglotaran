""" Glotaran Measured IRF"""

import numpy as np

from .irf import Irf


class MeasuredIrf(Irf):
    """A measured IRF."""

    def __init__(self, label):
        self._data = None
        super(MeasuredIrf, self).__init__(label)

    @property
    def data(self) -> np.ndarray:
        """Measured data."""
        if self._data is None:
            raise Exception(f"{self.label}: data not loaded")
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    def type_string(self):
        raise "Measured"

    def __str__(self):
        return f"### _{self.label}_\n* _Type_: {self.type_string()}\n"
