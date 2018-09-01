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
    'center_dispersion': {'type': List[str], 'default': []},
    'width_dispersion': {'type': List[str], 'default': []},
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
    def parameter(self, index):

        dist = (index - self.dispersion_center) if self.dispersion is not None else 0
        centers = self.center if isinstance(self.center, list) else [self.center]
        if len(self.center_dispersion) is not 0:
            for i, disp in enumerate(self.center_dispersion):
                centers = centers + disp * np.power(dist, i+1)

        widths = self.width if isinstance(self.width, list) else [self.width]
        if len(self.width_dispersion) is not 0:
            for i, disp in enumerate(self.width_dispersion):
                widths = widths + disp * np.power(dist, i+1)

        scale = self.scale if self.scale is not None else 1

        backsweep = 1 if self.backsweep else 0

        backsweep_period = self._irf.backsweep_period

        return centers, widths, scale, backsweep, backsweep_period



@glotaran_model_item_typed(types={
    'gaussian': IrfGaussian,
    'measured': IrfMeasured,
})
class Irf(object):
    """Represents an IRF."""
    pass
