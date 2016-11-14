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

        initial_concentration = dataset_descriptor.initial_concentration

        for m in dataset_descriptor.megacomplexes:
            cmplx = self._model.megacomplexes[m]
            c = self._construct_c_matrix_for_megacomplex(parameter,
                                                         cmplx,
                                                         initial_concentration,
                                                         times)

        return c

    def _construct_c_matrix_for_megacomplex(self, parameter, megacomplex,
                                            initial_concentration, times):

        # Combine K-Matrices of the megacomplex.

        model_k_matrix = self._get_combined_k_matrix(megacomplex)

        # Get K-Matrix array

        k_matrix = self._construct_k_matrix(parameter, model_k_matrix)

        # Get eigenvalues and vectors.

        eigenvalues, eigenvectors = \
            self._construct_k_matrix_eigen(parameter, k_matrix)

        # Calculate C Matrix

        C = np.empty((times.shape[0], eigenvalues.shape[0]),
                     dtype=np.float64)
        calculateC(C, eigenvalues, times)

        # Apply initial concentration vector if needed
        has_concentration_vector = \
            initial_concentration is not None
        if has_concentration_vector:

            # Get concentration vector

            concentration_vector = \
                self._model.initial_concentrations[
                    initial_concentration]\
                .parameter

            # Map to matrix

            # get compartment vector

            compartments = self._model.compartments

            # get compartment_map

            m = model_k_matrix.compartment_map

            # translate compartments to indices

            for i in range(len(m)):
                m[i] = compartments.index(m[i])

            # construct j_vector

            j_vector = [concentration_vector[i] for i in m]
            concentration_matrix = \
                self._construct_concentration_matrix(parameter,
                                                     j_vector,
                                                     eigenvectors)
            C = np.dot(C, concentration_matrix)

        return C

    def _get_combined_k_matrix(self, megacomplex):
        model_k_matrix = None
        for k_matrix_label in megacomplex.k_matrices:
            m = self._model.k_matrices[k_matrix_label]
            if model_k_matrix is None:
                model_k_matrix = m
            else:
                model_k_matrix = model_k_matrix.combine(m)
            return model_k_matrix

    def _construct_k_matrix_eigen(self, parameter, k_matrix):

        eigenvalues, eigenvectors = np.linalg.eig(k_matrix)
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

        return concentration_matrix

    def _construct_concentration_vector(self, parameter, megacomplex,
                                        dataset_descriptor):
        pass

    def _construct_k_matrix(self, parameter, model_k_matrix):
        k_matrix = model_k_matrix.asarray().astype(np.float64)
        k_matrix = self._parameter_map(parameter)(k_matrix)
        for i in range(k_matrix.shape[0]):
            k_matrix[i, i] = -np.sum(k_matrix[:, i])
        return k_matrix

    def e_matrix(self, **kwargs):
        dataset = self._model.datasets[kwargs['dataset']]
        amplitudes = kwargs["amplitudes"] if "amplitudes" in kwargs else None
        for megacomplex in dataset.megacomplexes:
            cmplx = self._model.megacomplexes[megacomplex]
            k_matrix = self._get_combined_k_matrix(cmplx)
            # E Matrix => channels X compartments
            nr_compartments = len(k_matrix.compartment_map)

            if amplitudes is None:
                E = np.full((1, nr_compartments), 1.0)
            else:
                E = np.empty((1, nr_compartments), dtype=np.float64)
                m = k_matrix.compartment_map
                compartments = self._model.compartments
                # translate compartments to indices

                for i in range(len(m)):
                    m[i] = compartments.index(m[i])

                # construct j_vector

                mapped_amps = [amplitudes[i] for i in m]

                for i in range(len(mapped_amps)):
                    E[1:i] = mapped_amps[i]

            break
        # get the
        return E

    def _parameter_map(self, parameter):
        def map_fun(i):
            if i != 0:
                i = parameter["p{}".format(int(i))]
            return i
        return np.vectorize(map_fun)
