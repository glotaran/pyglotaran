import numpy as np
from typing import List
from glotaran.model import glotaran_model_item, glotaran_model_item_typed


@glotaran_model_item(has_type=True)
class IrfMeasured:
    """A measured IRF."""

    _data = None

    @property
    def data(self) -> np.ndarray:
        """Measured data."""
        if self._data is None:
            raise Exception(f"{self.label}: data not loaded")
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value


@glotaran_model_item(attributes={
    'center': str,
    'width': str,
    'dispersion_center': {'type': str, 'default': None},
    'center_dispersion': {'type': List[str], 'default': None},
    'width_dispersion': {'type': List[str], 'default': None},
    'scale': {'type': str, 'default': None},
    'normalize': {'type': bool, 'default': False},
    'backsweep': {'type': bool, 'default': False},
    'backsweep_period': {'type': str, 'default': None},
}, has_type=True)
class IrfGaussian:
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameters
    ----------

    label:
        label of the irf
    center:
        one or more center of the irf as parameter indices
    width:
        one or more widths of the gaussian as parameter index
    center_dispersion:
        polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion:
        polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """
    pass


@glotaran_model_item_typed(types={
    'gaussian': IrfGaussian,
    'measured': IrfMeasured,
})
class Irf(object):
    """Represents an IRF."""
    pass
