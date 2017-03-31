import numpy as np
import scipy.linalg

from .c_matrix_cython.c_matrix_cython import CMatrixCython
from glotaran.fitmodel import parameter_map, parameter_idx_to_val, CMatrix


backend = CMatrixCython()


class KineticCMatrix(CMatrix):
    def __init__(self, x, dataset, model):

        super(KineticCMatrix, self).__init__(x, dataset, model)

        self._irf = None
        self._collect_irf(model)
        self._disp_center = model.dispersion_center

        self._k_matrices = []
        self._megacomplex_scaling = []
        self._collect_k_matrices(model)
        self._set_compartment_order()

        self._initial_concentrations = None
        self._collect_intital_concentration(model)

    def compartment_order(self):
        return self._compartment_order

    def shape(self):
        return (self.time().shape[0], len(self._compartment_order))

    def _set_compartment_order(self):
        compartment_order = [c for mat in self._k_matrices
                             for c in mat.compartment_map]

        self._compartment_order = list(set(compartment_order))

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
            scaling = self.dataset.megacomplex_scaling[mc.label] \
                if mc.label in self.dataset.megacomplex_scaling else None
            self._megacomplex_scaling.append(scaling)
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

        c_matrix = np.zeros(self.shape(), dtype=np.float64)

        for k_matrix, scale in self._k_matrices_and_scalings():
            tmp_c = self._calculate_for_k_matrix(k_matrix, parameter)

            if scale is not None:
                scale = np.prod(parameter_map(parameter)(scale))
                tmp_c *= scale

            for i in range(len(k_matrix.compartment_map)):
                target_idx = \
                    self.compartment_order().index(k_matrix.compartment_map[i])
                # TODO: implement scaling
                c_matrix[:, target_idx] += tmp_c[:, i]

        return self.scaling(parameter) * c_matrix

    def _k_matrices_and_scalings(self):
        for i in range(len(self._k_matrices)):
            yield self._k_matrices[i], self._megacomplex_scaling[i]

    def _calculate_for_k_matrix(self, k_matrix, parameter):

        # calculate k_matrix eigenvectos
        eigenvalues, eigenvectors = self._calculate_k_matrix_eigen(k_matrix,
                                                                   parameter)

        # get the time axis
        time = self.dataset.data.get_axis("time")

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

        for i in range(len(self.compartment_order)):
            comp = self.compartment_order[i]
            if comp in self.dataset.compartment_scalings:
                scale = self.dataset.compartment_scalings[comp]
                scale = np.prod(parameter_map(parameter)(scale))
                initial_concentrations[i] *= scale

        gamma = np.matmul(scipy.linalg.inv(eigenvectors),
                          initial_concentrations)

        concentration_matrix = np.empty(eigenvectors.shape,
                                        dtype=np.float64)
        for i in range(eigenvectors.shape[0]):
            concentration_matrix[i, :] = eigenvectors[:, i] * gamma[i]

        return np.dot(c_matrix, concentration_matrix)

    def time(self):
        return self.dataset.data.get_axis("time")

    def scaling(self, parameter):
        return parameter_idx_to_val(parameter, self.dataset.scaling) \
            if self.dataset.scaling is not None else 1
