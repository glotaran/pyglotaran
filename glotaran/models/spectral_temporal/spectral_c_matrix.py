import numpy as np

from glotaran.fitmodel import Matrix, parameter_idx_to_val

from .spectral_shape_gaussian import SpectralShapeGaussian


class SpectralCMatrix(Matrix):
    def __init__(self, x, dataset, model):
        super(SpectralCMatrix, self).__init__(x, dataset, model)

        self._shapes = {}
        self._collect_shapes(model)

        self._compartment_order = self.involved_compartments(model, dataset)

    def involved_compartments(self, model, dataset):
        cmplxs = [model.megacomplexes[c] for c in dataset.megacomplexes]
        kmats = [model.k_matrices[k] for cmplx in cmplxs
                 for k in cmplx.k_matrices]
        return list(set([c for kmat in kmats for c in kmat.compartment_map]))

    def _collect_shapes(self, model):

        for c, shape in self.dataset.shapes.items():
            self._shapes[c] = model.shapes[shape]

    def compartment_order(self):
        return self._compartment_order

    def shape(self):
        x = self.dataset.data.spectral_axis
        return (x.shape[0], len(self.compartment_order()))

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
