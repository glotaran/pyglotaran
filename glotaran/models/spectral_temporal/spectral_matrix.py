import numpy as np

from glotaran.fitmodel import Matrix

from .spectral_shape_gaussian import SpectralShapeGaussian


class SpectralMatrix(Matrix):
    def __init__(self, x, dataset, model):
        super(SpectralMatrix, self).__init__(x, dataset, model)

        self.shapes = {}
        self.collect_shapes()

    def collect_shapes(self):

        for c, shape in self.dataset.shapes.items():
            self.shapes[c] = self.model.shapes[shape]

    @property
    def compartment_order(self):
        cmplxs = [self.model.megacomplexes[c] for c in self.dataset.megacomplexes]
        kmats = [self.model.k_matrices[k] for cmplx in cmplxs
                 for k in cmplx.k_matrices]
        return list(set([c for kmat in kmats for c in kmat.compartment_map]))

    @property
    def shape(self):
        x = self.dataset.dataset.spectral_axis
        return (x.shape[0], len(self.compartment_order))

    def calculate(self, c_matrix, compartment_order, parameter):

        # We need the spectral shapes and axis to perform the calculations
        x = self.dataset.dataset.spectral_axis

        for (i, c) in enumerate(compartment_order):
            if c in self.shapes:
                c_matrix[:, i] = self._calculate_shape(parameter, self.shapes[c], x)
            else:
                # we use ones, so that if no shape is defined for the
                # compartment, the  amplitude is 1.0 by convention.
                c_matrix[:, i].fill(1.0)

    def _calculate_shape(self, parameter, shape, x):
        if isinstance(shape, SpectralShapeGaussian):
            amp = parameter.get(shape.amplitude)
            location = parameter.get(shape.location)
            width = parameter.get(shape.width)
            return amp * np.exp(-np.log(2) *
                                np.square(2 * (x - location)/width))
        raise ValueError(f"uknown shape '{type(shape)}'")
