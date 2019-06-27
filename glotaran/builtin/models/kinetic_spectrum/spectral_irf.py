import typing
import numpy as np

from glotaran.model import model_attribute
from glotaran.builtin.models.kinetic_image.irf import Irf, IrfGaussian
from glotaran.parameter import Parameter


@model_attribute(properties={
    'dispersion_center': {'type': Parameter, 'allow_none': True},
    'center_dispersion': {'type': typing.List[Parameter], 'default': []},
    'width_dispersion': {'type': typing.List[Parameter], 'default': []},
    'model_dispersion_with_wavenumber': {'type': bool, 'default': False},
    'coherent_artifact': {'type': bool, 'default': False},
    'coherent_artifact_order': {'type': int, 'allow_none': True},
}, has_type=True)
class IrfSpectralGaussian(IrfGaussian):
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
        centers, widths, scale, backsweep, backsweep_period = \
                super(IrfSpectralGaussian, self).parameter(index)

        if self.dispersion_center:
            dist = (1e3 / index - 1e3 / self.dispersion_center) \
                if self.model_dispersion_with_wavenumber else (index - self.dispersion_center)/100

        if len(self.center_dispersion) != 0:
            if self.dispersion_center is None:
                raise Exception(self, f'No dispersion center defined for irf "{self.label}"')
            for i, disp in enumerate(self.center_dispersion):
                centers += disp * np.power(dist, i+1)

        if len(self.width_dispersion) != 0:
            if self.dispersion_center is None:
                raise Exception(self, f'No dispersion center defined for irf "{self.label}"')
            for i, disp in enumerate(self.width_dispersion):
                widths = widths + disp * np.power(dist, i+1)

        return centers, widths, scale, backsweep, backsweep_period

    def calculate_coherent_artifact(self, index, axis):
        if not 1 <= self.coherent_artifact_order <= 3:
            raise Exception(self, "Coherent artifact order must be between in [1,3]")

        center, width, scale, _, _ = self.parameter(index)

        matrix = np.zeros((axis.size, self.coherent_artifact_order), dtype=np.float64)
        for i in range(len(center)):

            irf = np.exp(-1 * (axis - center[i])**2 / (2 * width[i]**2))
            matrix[:, 0] = irf * scale[i]

            if self.coherent_artifact_order > 1:
                matrix[:, 1] = irf * (center[i] - axis) / width[i]**2

            if self.coherent_artifact_order > 2:
                matrix[:, 2] = \
                    irf * (center[i]**2 - width[i]**2 - 2 * center[i] * axis + axis**2) \
                    / width[i]**4

        return self.clp_labels(), matrix

    def clp_labels(self):
        return [f'{self.label}_coherent_artifact_{i}'
                for i in range(1, self.coherent_artifact_order + 1)]

    def calculate_dispersion(self, axis):
        dispersion = []
        for index in axis:
            center, _, _, _, _ = self.parameter(index)
            dispersion.append(center)
        return np.asarray(dispersion).T


Irf.add_type('spectral-gaussian', IrfSpectralGaussian)
