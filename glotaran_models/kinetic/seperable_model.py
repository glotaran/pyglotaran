from glotaran_core.fitting.variable_projection import SeperableModel
from lmfit import Parameters
import numpy as np
import scipy.linalg
from c_matrix import calculateC
from c_matrix_gaussian_irf import calculateCMultiGaussian
from glotaran_core.model import BoundConstraint, FixedConstraint


class KineticSeperableModel(SeperableModel):
    def __init__(self, model):
        self._model = model
        self._prepare_parameter()

    def _prepare_parameter(self):
        self._fit_params = Parameters()

        # get fixed param indices
        fixed = []
        bound = []
        relations = []

        if self._model.relations is not None:
            relations = [r.parameter for r in self._model.relations]

        if self._model.parameter_constraints is not None:
            i = 0
            for constraint in self._model.parameter_constraints:
                if isinstance(constraint, FixedConstraint):
                    for p in constraint.parameter:
                        fixed.append(p)
                elif isinstance(constraint, BoundConstraint):
                    bound.append((i, constraint.parameter))
                i += 1

        for p in self._model.parameter:
            if not p.value == 'NaN':
                vary = p.index not in fixed
                min, max = None, None
                expr = None
                val = p.value
                for i in range(len(bound)):
                    if p.index in bound[i][1]:
                        b = self._model.relations[bound[i][0]]
                        if b.min != 'NaN':
                            min = b.min
                        if b.max != 'NaN':
                            max = b.max

                if p.index in relations:
                    r = self._model.relations[relations.index(p.index)]
                    vary = False
                    val = None
                    first = True
                    expr = ''
                    for target in r.to:
                        if not first:
                            expr += "+"
                        first = False
                        if target == 'const':
                            expr += "{}".format(r.to[target])
                        else:
                            expr += "p{}*{}".format(target, r.to[target])

                self._fit_params.add("p{}".format(p.index), val,
                                     vary=vary, min=min, max=max, expr=expr)

    def c_matrix(self, parameter, *times, **kwargs):
        for dataset in self._model.datasets:
            desc = self._model.datasets[dataset]
            c = self._construct_c_matrix_for_dataset(parameter, desc,
                                                     np.asarray(times))
            break
        return c

    def get_initial_fitting_parameter(self):
        return self._fit_params

    def _construct_c_matrix_for_dataset(self, parameter, dataset_descriptor,
                                        times):

        initial_concentration = dataset_descriptor.initial_concentration
        irf = dataset_descriptor.irf
        c_matrix = None
        for m in dataset_descriptor.megacomplexes:
            cmplx = self._model.megacomplexes[m]
            tmp = \
                self._construct_c_matrix_for_megacomplex(parameter,
                                                         cmplx,
                                                         initial_concentration,
                                                         irf,
                                                         times)
            if c_matrix is None:
                c_matrix = tmp
            else:
                if c_matrix.shape[1] > tmp.shape[1]:
                    for i in range(tmp.shape[1]):
                        c_matrix[:, i] = tmp[:, i] + c_matrix[:, i]
                else:
                    for i in range(c_matrix.shape[1]):
                        c_matrix[:, i] = tmp[:, i] + c_matrix[:, i]

        return c_matrix

    def _construct_c_matrix_for_megacomplex(self, parameter, megacomplex,
                                            initial_concentration, irf, times):

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

        has_irf = irf is not None

        if has_irf:
            center, width, scale = self._get_irf_parameter(parameter, irf)
            calculateCMultiGaussian(C, eigenvalues, times, center, width,
                                    scale)
        else:
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

    def _get_irf_parameter(self, parameter, irf):

        irf = self._model.irfs[irf]

        center = self._parameter_map(parameter)(np.asarray(irf.center))
        width = self._parameter_map(parameter)(np.asarray(irf.width))
        if width.shape[0] is not center.shape[0]:
            width = np.fill(center.shape, width[0])
        if len(irf.scale) is 0:
            scale = np.ones(center.shape)
        else:
            scale = self._parameter_map(parameter)(np.asarray(irf.scale))

        return center, width, scale

    def e_matrix(self, **kwargs):
        dataset = self._model.datasets[kwargs['dataset']]
        amplitudes = kwargs["amplitudes"] if "amplitudes" in kwargs else None
        e = None
        for megacomplex in dataset.megacomplexes:
            cmplx = self._model.megacomplexes[megacomplex]
            k_matrix = self._get_combined_k_matrix(cmplx)
            # E Matrix => channels X compartments
            nr_compartments = len(k_matrix.compartment_map)

            if amplitudes is None:
                tmp = np.full((1, nr_compartments), 1.0)
            else:
                tmp = np.empty((1, nr_compartments), dtype=np.float64)
                m = k_matrix.compartment_map
                compartments = self._model.compartments
                # translate compartments to indices

                for i in range(len(m)):
                    m[i] = compartments.index(m[i])

                mapped_amps = [amplitudes[i] for i in m]

                for i in range(len(mapped_amps)):
                    tmp[:, i] = mapped_amps[i]
                print(tmp)
            if e is None:
                e = tmp
            else:
                if e.shape[1] > tmp.shape[1]:
                    for i in range(tmp.shape[1]):
                        e[:, i] = tmp[:, i] + e[:, i]
                else:
                    for i in range(e.shape[1]):
                        tmp[:, i] = tmp[:, i] + e[:, i]
                        e = tmp

            break
        # get the

        return e

    def _parameter_map(self, parameter):
        def map_fun(i):
            if i != 0:
                i = parameter["p{}".format(int(i))]
            return i
        return np.vectorize(map_fun)
