import numpy as np

from glotaran_core.fitting.variable_projection import SeperableModelResult


class KineticSeperableModelResult(SeperableModelResult):
    _left_singular_values = None
    _singular_values = None
    _right_singular_values = None

    def coefficients(self, *args, **kwargs):
        dataset = self.model._model.datasets[kwargs['dataset']]

        for megacomplex in dataset.megacomplexes:
            cmplx = self.model._model.megacomplexes[megacomplex]
            k_matrix = self.model._get_combined_k_matrix(cmplx)
            m = k_matrix.compartment_map
            compartments = self.model._model.compartments
            for i in range(len(m)):
                m[i] = compartments.index(m[i])
            e_matrix = self.e_matrix(*args, **kwargs)
            mapped_e_matrix = np.empty(e_matrix.shape, e_matrix.dtype)
            for i in range(len(m)):
                mapped_e_matrix[m[i], :] = e_matrix[i]
            return mapped_e_matrix

    def spectra(self, *args, **kwargs):
        return self.coefficients(*args, **kwargs)

    def normalized_spectra(self, *args, **kwargs):
        spectra = self.spectra(*args, **kwargs)
        for i in range(spectra.shape[0]):
            spectra[i, :] = spectra[i, :] / np.abs(spectra[i, :]).max()
        return spectra

    def svd(self, *args, **kwargs):
        reconstructed = self.eval(*args, **kwargs)
        lsvd, svals, rsvd = np.linalg.svd(reconstructed)
        return lsvd, svals, rsvd
