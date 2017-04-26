import numpy as np

from glotaran.fitmodel import CMatrix, parameter_idx_to_val

from .spectral_shape_gaussian import SpectralShapeGaussian


class SpectralCMatrix(CMatrix):
    def __init__(self, x, dataset, model):
        super(SpectralCMatrix, self).__init__(x, dataset, model)

        self._shapes = []
        self._collect_shapes(model)

        self._compartment_order = model.compartments

    def _collect_shapes(self, model):

        # We create a shape for every compartment. If we find a shape for a
        # compartment in the descriptor, we set the index of the compartment in
        # our shape vector to the shape
        self._shapes = list([None for _ in model.compartments])
        for c, shape in self.dataset.shapes.items():
            idx = model.compartments.index(c)
            self._shapes[idx] = model.shapes[shape]

    def compartment_order(self):
        return self._compartment_order

    def shape(self):
        shapes = self._shapes
        x = self.dataset.data.spectral_axis
        return (x.shape[0], len(shapes))

    def calculate(self, parameter):
        shapes = self._shapes
        x = self.dataset.data.spectral_axis

        # we use ones, so that if no shape is defined for the compartment, its
        # amplitude is 1.0 by convention.
        mat = np.ones(self.shape(), np.float64)
        for i in range(len(shapes)):
            if shapes[i] is not None:
                mat[:, i] = self._calculate_shape(parameter, shapes[i], x)
        return mat

    def _calculate_shape(self, parameter, shape, x):
        if isinstance(shape, SpectralShapeGaussian):
            amp = parameter_idx_to_val(parameter, shape.amplitude)
            location = parameter_idx_to_val(parameter, shape.location)
            width = parameter_idx_to_val(parameter, shape.width)
            return amp * np.exp(-np.log(2) *
                                np.square(2 * (x - location)/width))
