"""This package contains irf items."""

from typing import List
import numpy as np

from glotaran.model import model_item, model_item_typed

class IrfException(Exception):
    def __init__(self, irf, msg):
        self.irf = irf.label
        self.msg = msg

    def __str__(self):
        return f"Irf '{self.irf.label}' error: {self.msg}"


@model_item(attributes={
    'irfdata': {'type': np.ndarray, 'default': None},
}, has_type=True)
class IrfMeasured:
    """A measured IRF."""


@model_item(attributes={
    'center': str,
    'width': str,
    'dispersion_center': {'type': str, 'default': None},
    'center_dispersion': {'type': List[str], 'default': []},
    'width_dispersion': {'type': List[str], 'default': []},
    'scale': {'type': str, 'default': None},
    'normalize': {'type': bool, 'default': False},
    'backsweep': {'type': bool, 'default': False},
    'backsweep_period': {'type': str, 'default': None},
    'coherent_artifact': {'type': bool, 'default': False},
    'coherent_artifact_order': {'type': int, 'default': 1},
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

        centers = self.center if isinstance(self.center, list) else [self.center]
        if len(self.center_dispersion) is not 0:
            if self.dispersion_center is None:
                raise IrfException(self, f'No dispersion center defined for irf "{self.label}"')
            dist = (index - self.dispersion_center)/100
            for i, disp in enumerate(self.center_dispersion):
                centers += disp * np.power(dist, i+1)

        widths = self.width if isinstance(self.width, list) else [self.width]
        if len(self.width_dispersion) is not 0:
            if self.dispersion_center is None:
                raise IrfException(self, f'No dispersion center defined for irf "{self.label}"')
            dist = (index - self.dispersion_center)/100
            for i, disp in enumerate(self.width_dispersion):
                widths = widths + disp * np.power(dist, i+1)

        len_centers = len(centers)
        len_widths = len(widths)
        if not len_centers == len_widths:
            if not min(len_centers, len_widths) == 1:
                raise IrfException(f'len(centers) ({len_centers}) not equal '
                                   f'len(widths) ({len_widths}) none of is 1.')
            if len_centers == 1:
                centers = [centers[0] for _ in range(len_widths)]
                len_centers = len_widths
            else:
                widths = [widths[0] for _ in range(len_centers)]
                len_widths = len_centers

        scale = self.scale if self.scale is not None else [1 for _ in centers]
        scale = scale if isinstance(scale, list) else [scale]
        if self.normalize:
            scale /= np.sqrt(2 * np.pi * np.asarray(widths)**2)

        backsweep = 1 if self.backsweep else 0

        backsweep_period = self.backsweep_period if backsweep else 0

        return centers, widths, scale, backsweep, backsweep_period

    def calculate_coherent_artifact(self, index, axis):
        if not 1 <= self.coherent_artifact_order <= 3:
            raise IrfException(self, "Coherent artifact order must be between in [1,3]")

        center, width, scale, _, _ = self.parameter(index)

        matrix = np.zeros((axis.size, self.coherent_artifact_order), dtype=np.float64)

        irf = np.exp(-1 * (axis - center)**2 / (2 * width**2))
        matrix[:, 0] = irf

        if self.coherent_artifact_order > 1:
            matrix[:, 1] = irf * (center - axis) / width**2

        if self.coherent_artifact_order > 2:
            matrix[:, 2] = irf * (center**2 - width**2 - 2 * center * axis + axis**2) / width**4

        matrix *= scale

        return self.clp_labels(), matrix

    def clp_labels(self):
        return [f'{self.label}_coherent_artifact_{i}'
                for i in range(1, self.coherent_artifact_order + 1)]

    def calculate(self, index, axis):
        center, width, scale, _, _ = self.parameter(index)
        irf = scale[0] * np.exp(-1 * (axis - center[0])**2 / (2 * width[0]**2))
        if len(center) > 1:
            for i in range(1, len(center)):
                irf += scale[i] * np.exp(-1 * (axis - center[i])**2 / (2 * width[i]**2))
        return irf

    def calculate_dispersion(self, axis):
        dispersion = []
        for index in axis:
            center, _, _, _, _ = self.parameter(index)
            dispersion.append(center)
        return np.asarray(dispersion).T


@model_item_typed(types={
    'gaussian': IrfGaussian,
    'measured': IrfMeasured,
})
class Irf(object):
    """Represents an IRF."""
