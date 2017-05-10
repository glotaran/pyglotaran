import numpy as np

from glotaran.fitmodel import CMatrix, parameter_idx_to_val

from .spectral_shape_gaussian import SpectralShapeGaussian


class SpectralCMatrix(CMatrix):
    def __init__(self, x, dataset, model):
        super(SpectralCMatrix, self).__init__(x, dataset, model)

        self._shapes = {}
        self._collect_shapes(model)

        if len(self.dataset.shapes) is 0:
            self._compartment_order = model.compartments
        else:
            self._compartment_order = [c for c in model.compartments if c in self.dataset.shapes]

    def _collect_shapes(self, model):

        for c, shape in self.dataset.shapes.items():
            self._shapes[c] = model.shapes[shape]

    def compartment_order(self):
        return self._compartment_order

    def shape(self):
        shapes = self._shapes
        x = self.dataset.data.spectral_axis
        return (x.shape[0], len(shapes))

    def calculate(self, c_matrix, compartment_order, parameter):
        shapes = self._shapes
        x = self.dataset.data.spectral_axis

        # we use ones, so that if no shape is defined for the compartment, its
        # amplitude is 1.0 by convention.
        i = 0
        for c in compartment_order:
            if c in shapes:
                c_matrix[:, i] = self._calculate_shape(parameter, shapes[c], x)
            else:
                c_matrix[:, i].fill(1.0)
            i += 1

    def _calculate_shape(self, parameter, shape, x):
        if isinstance(shape, SpectralShapeGaussian):
            amp = parameter_idx_to_val(parameter, shape.amplitude)
            location = parameter_idx_to_val(parameter, shape.location)
            width = parameter_idx_to_val(parameter, shape.width)
            return amp * np.exp(-np.log(2) *
                                np.square(2 * (x - location)/width))
