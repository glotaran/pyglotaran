from collections import OrderedDict
import numpy as np
import scipy.linalg

from .c_matrix_cython.c_matrix_cython import CMatrixCython


backend = CMatrixCython()


class CMatrixGenerator(object):
    def __init__(self):
        self._groups = OrderedDict()

    @classmethod
    def for_model(cls, model, xtol=0.5):
        gen = cls()
        gen._init_groups_for_model(model, xtol)
        return gen

    @classmethod
    def for_dataset(cls, model, dataset):
        gen = cls()
        data = model.datasets[dataset]
        gen._add_dataset_to_group(model, data, 0)
        return gen

    def _init_groups_for_model(self, model, xtol):
        for _, dataset in model.datasets.items():
            self._add_dataset_to_group(model, dataset, xtol)

    def _add_dataset_to_group(self, model, dataset, xtol):
            for matrix in [CMatrix(x, dataset, model) for x in
                           dataset.data.independent_axies.get(0)]:
                self._add_c_matrix_to_group(matrix, xtol)

    def _add_c_matrix_to_group(self, matrix, xtol):
                if matrix.x in self._groups:
                    self._groups[matrix.x].add_cmatrix(matrix)
                elif any(abs(matrix.x-val) < xtol for val in self._groups):
                    idx = [val for val in self._groups if abs(matrix.x-val) <
                           xtol][0]
                    self._groups[idx].add_cmatrix(matrix)
                else:
                    self._groups[matrix.x] = CMatrixGroup(matrix)

    def groups(self):
        for _, group in self._groups.items():
            yield group

    def calculate(self, parameter):
        return [group.calculate(parameter) for group in self.groups()]

    def create_dataset_group(self):

        dataset_group = []
        for _, group in self._groups.items():
            slices = []
            for mat in group.c_matrices:
                x = np.where(mat.dataset.data.independent_axies.get(0) ==
                             mat.x)[0]
                slices.append(mat.dataset.data.data[:, x])
            slice = np.append([], slices)
            dataset_group.append(slice)

        return dataset_group


class CMatrixGroup(object):
    def __init__(self, c_matrix):
        self.id = c_matrix.x
        self.c_matrices = [c_matrix]

    def add_cmatrix(self, c_matrix):
        self.c_matrices.append(c_matrix)

    def calculate(self, parameter):

        self._set_compartment_order()

        c_matrix = np.zeros((self.time().shape[0],
                             len(self.compartment_order)),
                            dtype=np.float64)

        t_idx = 0

        for cmat in self.c_matrices:
            tmp_c = cmat.calculate(parameter)

            n_t = t_idx + len(cmat.time())
            for i in range(len(cmat.compartment_order)):
                target_idx = \
                    self.compartment_order.index(cmat.compartment_order[i])
                c_matrix[t_idx:n_t, target_idx] = tmp_c[:, i]
            t_idx = n_t

        return c_matrix

    def time(self):
        return np.asarray([t for cmat in self.c_matrices for t in cmat.time()])

    def _set_compartment_order(self):
        compartment_order = [c for cmat in self.c_matrices
                             for c in cmat.compartment_order]

        self.compartment_order = list(set(compartment_order))


class CMatrix(object):
    def __init__(self, x, dataset, model):
        self.x = x
        self.dataset = dataset

        self._irf = None
        self._collect_irf(model)
        self._disp_center = model.dispersion_center

        self._k_matrices = []
        self._collect_k_matrices(model)
        self._set_compartment_order()

        self._initial_concentrations = None
        self._collect_intital_concentration(model)

    def _set_compartment_order(self):
        compartment_order = [c for mat in self._k_matrices
                             for c in mat.compartment_map]

        self.compartment_order = list(set(compartment_order))

    def _collect_irf(self, model):
        if self.dataset.irf is None:
            return
        self._irf = model.irfs[self.dataset.irf]

    def _collect_k_matrices(self, model):
        for mc in [model.megacomplexes[mc] for mc in
                   self.dataset.megacomplexes]:
            model_k_matrix = None
            for k_matrix_label in mc.k_matrices:
                m = model.k_matrices[k_matrix_label]
                if model_k_matrix is None:
                    model_k_matrix = m
                else:
                    model_k_matrix = model_k_matrix.combine(m)
        self._k_matrices.append(model_k_matrix)

    def _collect_intital_concentration(self, model):

        if self.dataset.initial_concentration is None:
            return
        initial_concentrations = \
            model.initial_concentration[self.dataset.initial_concentration]

        # The initial concentration vector has an element for each compartment
        # declared in the model. The current C Matrix must not necessary invole
        # all compartments, as well as the order of compartments can be
        # different. Thus we shrink and reorder the concentration.

        all_cmps = model.compartments

        self._initial_concentrations = \
            [initial_concentrations[all_cmps.index(c)]
             for c in self.compartment_order]

    def calculate(self, parameter):

        c_matrix = np.zeros((self.time().shape[0],
                             len(self.compartment_order)),
                            dtype=np.float64)

        for k_matrix in self._k_matrices:
            tmp_c = self._calculate_for_k_matrix(k_matrix, parameter)

            for i in range(len(k_matrix.compartment_map)):
                target_idx = \
                    self.compartment_order.index(k_matrix.compartment_map[i])
                # TODO: implement scaling
                c_matrix[:, target_idx] += tmp_c[:, i]

        return c_matrix

    def _calculate_for_k_matrix(self, k_matrix, parameter):

        # calculate k_matrix eigenvectos
        eigenvalues, eigenvectors = self._calculate_k_matrix_eigen(k_matrix,
                                                                   parameter)

        # get the time axis
        time = self.dataset.data.independent_axies.get(1)

        # allocate C matrix
        # TODO: do this earlier

        c_matrix = np.empty((time.shape[0], eigenvalues.shape[0]),
                            dtype=np.float64)

        if self._irf is None:
            backend.c_matrix(c_matrix, eigenvalues, time)
        else:
            centers, widths, scale = self._calculate_irf_parameter(parameter)
            backend.c_matrix_gaussian_irf(c_matrix, eigenvalues,
                                          time,
                                          centers, widths,
                                          scale)

        if self._initial_concentrations is not None:
            c_matrix = self._apply_initial_concentration_vector(c_matrix,
                                                                eigenvectors,
                                                                parameter)

        return c_matrix

    def _calculate_k_matrix_eigen(self, k_matrix, parameter):

        # convert k_matrix to np.array and replace indices with actual
        # parameters
        k_matrix = k_matrix.asarray().astype(np.float64)
        k_matrix = parameter_map(parameter)(k_matrix)

        # construct the full k matrix matrix
        for i in range(k_matrix.shape[0]):
            k_matrix[i, i] = -np.sum(k_matrix[:, i])

        # get the eigenvectors and values
        eigenvalues, eigenvectors = np.linalg.eig(k_matrix)
        return(eigenvalues, eigenvectors)

    def _calculate_irf_parameter(self, parameter):

        centers = parameter_map(parameter)(np.asarray(self._irf.center))
        widths = parameter_map(parameter)(np.asarray(self._irf.width))

        center_dispersion = \
            parameter_map(parameter)(np.asarray(self._irf.center_dispersion)) \
            if len(self._irf.center_dispersion) is not 0 else []

        width_dispersion = \
            parameter_map(parameter)(np.asarray(self._irf.width_dispersion)) \
            if len(self._irf.width_dispersion) is not 0 else []

        dist = (self.x - self._disp_center)/100
        if len(center_dispersion) is not 0:
            for i in range(len(center_dispersion)):
                centers = centers + center_dispersion[i] * np.power(dist, i+1)

        if len(width_dispersion) is not 0:
            for i in range(len(width_dispersion)):
                widths = widths + width_dispersion[i] * np.power(dist, i+1)

        if len(self._irf.scale) is 0:
            scale = np.ones(centers.shape)
        else:
            scale = parameter_map(parameter)(np.asarray(self._irf.scale))

        return centers, widths, scale

    def _apply_initial_concentration_vector(self, c_matrix, eigenvectors,
                                            parameter):

        initial_concentrations = \
            parameter_map(parameter)(self._initial_concentrations)

        gamma = np.matmul(scipy.linalg.inv(eigenvectors),
                          initial_concentrations)

        concentration_matrix = np.empty(eigenvectors.shape,
                                        dtype=np.float64)
        for i in range(eigenvectors.shape[0]):
            concentration_matrix[i, :] = eigenvectors[:, i] * gamma[i]

        return np.dot(c_matrix, concentration_matrix)

    def time(self):
        return self.dataset.data.independent_axies.get(1)


def parameter_map(parameter):
    def map_fun(i):
        if i != 0:
            i = parameter["p{}".format(int(i))]
        return i
    return np.vectorize(map_fun)
