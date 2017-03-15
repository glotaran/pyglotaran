from lmfit import Parameters
import numpy as np
import scipy.linalg

from lmfit_varpro import SeparableModel
from glotaran.model import BoundConstraint, FixedConstraint

from c_matrix import calculateC
from .c_matrix_cython.c_matrix_cython import CMatrixCython
#from .c_matrix_python import CMatrixPython
#from .c_matrix_opencl.c_matrix_opencl import CMatrixOpenCL

from .result import KineticSeparableModelResult


class KineticSeparableModel(SeparableModel):
    def __init__(self, model):
        self._model = model
        self._prepare_parameter()
        self._c_matrix_backend = CMatrixCython()

    def data(self, **kwargs):
        data = ()
        for lbl, dataset in self._model.datasets.items():
            data = data + (dataset.data.data,)
        return data

    def fit(self, initial_parameter, *args, **kwargs):
        result = KineticSeparableModelResult(self, initial_parameter, *args,
                                             **kwargs)
        result.fit(initial_parameter, *args, **kwargs)
        return result

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

    def c_matrix(self, parameter, *args, **kwargs):
        for dataset in self._model.datasets:
            desc = self._model.datasets[dataset]
            c = self._construct_c_matrix_for_dataset(parameter, desc)
            break
        return c

    def get_initial_fitting_parameter(self):
        return self._fit_params

    def _construct_c_matrix_for_dataset(self, parameter, dataset_descriptor):
        axies = dataset_descriptor.data.independent_axies
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
                                                         axies)
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

    def _construct_c_matrix_for_megacomplex(self,
                                            parameter,
                                            megacomplex,
                                            initial_concentration,
                                            irf,
                                            axies):
        # Combine K-Matrices of the megacomplex.

        model_k_matrix = self._get_combined_k_matrix(megacomplex)

        # Get K-Matrix array

        k_matrix = self._construct_k_matrix(parameter, model_k_matrix)

        # Get eigenvalues and vectors.

        eigenvalues, eigenvectors = \
            self._construct_k_matrix_eigen(parameter, k_matrix)

        # Calculate C Matrix

        x = axies.get(0)
        times = axies.get(1)

        C = np.empty((x.shape[0], times.shape[0], eigenvalues.shape[0]),
                     dtype=np.float64)

        #  num_threads = multiprocessing.cpu_count()

        if irf is not None:
            centers, widths, scale = self._get_irf_parameter(parameter, irf,
                                                             x)
            self._c_matrix_backend.c_matrix_gaussian_irf(C, eigenvalues,
                                                         times,
                                                         centers, widths,
                                    scale)
        else:
            calculateC(C[0, :, :], eigenvalues, times)
            for i in range(1, C.shape[0]):
                C[i, :, :] = C[0, :, :]

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
        #  idx = eigenvalues.argsort()[::-1]
        #  eigenvalues = eigenvalues[idx]
        #  eigenvectors = eigenvectors[:, idx]

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

    def _get_irf_parameter(self, parameter, irf, x):

        irf = self._model.irfs[irf]

        center = self._parameter_map(parameter)(np.asarray(irf.center))
        centers = np.asarray([center for _ in x])

        center_dispersion = \
            self._parameter_map(parameter)(np.asarray(irf.center_dispersion)) \
            if len(irf.center_dispersion) is not 0 else None
        width = self._parameter_map(parameter)(np.asarray(irf.width))
        width_dispersion = \
            self._parameter_map(parameter)(np.asarray(irf.width_dispersion)) \
            if len(irf.width_dispersion) is not 0 else None

        if center_dispersion is not None or width_dispersion is not None:
            dist = (x - x[0])/100

        if center_dispersion is not None:
            for i in range(len(center_dispersion)):
                centers = centers + center_dispersion[i] * np.power(dist, i+1)

        if width.shape[0] is not center.shape[0]:
            width = np.full(center.shape, width[0])

        widths = np.asarray([width for _ in x])

        if width_dispersion is not None:
            for i in range(len(width_dispersion)):
                widths = widths + width_dispersion[i] * np.power(dist, i+1)

        if len(irf.scale) is 0:
            scale = np.ones(center.shape)
        else:
            scale = self._parameter_map(parameter)(np.asarray(irf.scale))

        return centers, widths, scale

    def e_matrix(self, **kwargs):
        dataset = self._model.datasets[kwargs['dataset']]
        amplitudes = kwargs["amplitudes"] if "amplitudes" in kwargs else None
        locations = kwargs["locations"] if "locations" in kwargs else None
        delta = kwargs["delta"] if "delta" in kwargs else None
        x = dataset.data.independent_axies.get(0)
        e = None
        for megacomplex in dataset.megacomplexes:
            cmplx = self._model.megacomplexes[megacomplex]
            k_matrix = self._get_combined_k_matrix(cmplx)
            #  E Matrix => channels X compartments
            nr_compartments = len(k_matrix.compartment_map)

            if amplitudes is None:
                tmp = np.full((len(x), nr_compartments), 1.0)
            else:
                tmp = np.empty((len(x), nr_compartments), dtype=np.float64)
                m = k_matrix.compartment_map
                compartments = self._model.compartments
                # translate compartments to indices

                for i in range(len(m)):
                    m[i] = compartments.index(m[i])

                mapped_amps = [amplitudes[i] for i in m]

                for i in range(len(mapped_amps)):
                    for j in range(len(x)):
                        if locations is None or delta is None:
                            tmp[j, i] = mapped_amps[i]
                        else:
                            mapped_locs = [locations[i] for i in m]
                            mapped_delta = [delta[i] for i in m]
                            tmp[:, i] = mapped_amps[i] * np.exp(
                                -np.log(2) * np.square(
                                    2 * (x - mapped_locs[i])/mapped_delta[i]
                                )
                            )

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

    def coefficients(self, *args, **kwargs):
        dataset = self._model.datasets[kwargs['dataset']]

        for megacomplex in dataset.megacomplexes:
            cmplx = self._model.megacomplexes[megacomplex]
            k_matrix = self._get_combined_k_matrix(cmplx)
            m = k_matrix.compartment_map
            compartments = self._model.compartments
            for i in range(len(m)):
                m[i] = compartments.index(m[i])
            e_matrix = self.e_matrix(*args, **kwargs)
            mapped_e_matrix = np.empty(e_matrix.shape, e_matrix.dtype)
            for i in range(len(m)):
                mapped_e_matrix[:, m[i]] = e_matrix[:, i]
            return mapped_e_matrix

    def _parameter_map(self, parameter):
        def map_fun(i):
            if i != 0:
                i = parameter["p{}".format(int(i))]
            return i
        return np.vectorize(map_fun)
