from glotaran_core.fitting.variable_projection import SeperableModel
from lmfit import Parameters
import numpy as np
import scipy.linalg
from c_matrix import calculateC


class KineticSeperableModel(SeperableModel):
    def __init__(self, model):
        self._model = model

    def get_nr_of_compartements(self):
        return [self._model.k_matrices[k].matrix for k in
                self._model.k_matrices][0].shape[0]

    def get_kinetic_parameter(self):
        kinetic_parameter = []
        for i in range(self.get_nr_of_compartements()):
            par_index = [self._model.k_matrices[k].matrix for k in
                         self._model.k_matrices][0][i, i]
            kinetic_parameter.append(par_index)
        return kinetic_parameter

    def c_matrix(self, parameter, *times, **kwargs):
        for dataset in self._model.datasets:
            desc = self._model.datasets[dataset]
            c = self._construct_c_matrix_for_dataset(parameter, desc,
                                                     np.asarray(times))
            break
        return c

    def get_initial_fitting_parameter(self):
        params = Parameters()
        for p in self._model.parameter:
            params.add("p{}".format(p.index), p.value)
        return params

    def _construct_c_matrix_for_dataset(self, parameter, dataset_descriptor,
                                        times):

        for m in dataset_descriptor.megacomplexes:
            c = self._construct_c_matrix_for_megacomplex(parameter, m, None,
                                                         times)

        return c

    def _construct_c_matrix_for_megacomplex(self, parameter, megacomplex,
                                            initial_concentration, times):

        model_k_matrix_label = \
                self._model.megacomplexes[megacomplex].k_matrices[0]
        model_k_matrix = self._model.k_matrices[model_k_matrix_label].matrix
        model_k_matrix = model_k_matrix.toarray().astype(np.float64)
        eigenvalues, eigenvectors = \
            self._construct_k_matrix_eigen(parameter, model_k_matrix)

        has_concentration_vector = \
            initial_concentration is not None

        C = np.empty((times.shape[0], eigenvalues.shape[0]),
                     dtype=np.float64)

        calculateC(C, eigenvalues, times)

        if has_concentration_vector:
            concentration_vector = \
                self._model.initial_concentrations[
                    initial_concentration]\
                .parameter
            concentration_matrix = \
                self._construct_concentration_matrix(parameter,
                                                     concentration_vector,
                                                     eigenvectors)
            C = np.dot(C, concentration_matrix)

        return C

    def _construct_k_matrix_eigen(self, parameter, model_k_matrix):

        k_matrix = self._construct_k_matrix(parameter, model_k_matrix)

        eigenvalues, eigenvectors = np.linalg.eig(k_matrix.astype(np.float64))
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return(eigenvalues, eigenvectors)

    def _construct_concentration_matrix(self, parameter, concentration_vector,
                                        k_matrix_eigenvectors):

        j_vector = self._parameter_map(parameter)(concentration_vector)
        gamma = np.matmul(scipy.linalg.inv(k_matrix_eigenvectors), j_vector)

        concentration_matrix = np.empty(k_matrix_eigenvectors.shape,
                                        dtype=np.float64)
        for i in range(k_matrix_eigenvectors.shape[0]):
            concentration_matrix[i, :] = k_matrix_eigenvectors[:, i] * gamma[i]

    def _construct_k_matrix(self, parameter, model_k_matrix):
        k_matrix = self._parameter_map(parameter)(model_k_matrix)

        for i in range(k_matrix.shape[0]):
            k_matrix[i, i] = -np.sum(k_matrix[:, i])
        return k_matrix

    def _parameter_map(self, parameter):
        def map_fun(i):
            if i != 0:
                i = parameter["p{}".format(int(i))]
            return i
        return np.vectorize(map_fun)
