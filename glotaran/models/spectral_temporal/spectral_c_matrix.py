import numpy as np

from glotaran.fitmodel import CMatrix

from .spectral_shape_gaussian import SpectralShapeGaussian


class SpectralCMatrix(CMatrix):
    def __init__(self, x, dataset, model):
        super(SpectralCMatrix, self).__init__(x, dataset, model)

        self._shapes = []
        self._collect_shapes(model)

        self._compartment_order = model.compartments

    def _collect_shapes(self, model):
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

        mat = np.ones(self.shape(), np.float64)

        for i in range(len(shapes)):
            if shapes[i] is not None:
                mat[:, i] = self._calcuate_shape(shapes[i], x)
        return mat

    def _calculate_shape(self, shape, x):
        if isinstance(shape, SpectralShapeGaussian):
            return shape.amplitude * np.exp(
                                -np.log(2) * np.square(
                                    2 * (x - shape.location)/shape.width
                                )
                            )
