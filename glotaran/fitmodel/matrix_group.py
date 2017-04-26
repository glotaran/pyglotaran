import numpy as np


class MatrixGroup(object):
    def __init__(self, c_matrix):
        self.id = c_matrix.x
        self.c_matrices = [c_matrix]

    def add_cmatrix(self, c_matrix):
        self.c_matrices.append(c_matrix)

    def calculate(self, parameter):

        self._set_compartment_order()

        c_matrix = np.zeros(self.shape(), dtype=np.float64)

        t_idx = 0

        for cmat in self.c_matrices:
            tmp_c = cmat.calculate(parameter)

            n_t = t_idx + cmat.shape()[0]
            for i in range(cmat.shape()[1]):
                target_idx = \
                    self.compartment_order.index(cmat.compartment_order()[i])
                c_matrix[t_idx:n_t, target_idx] = tmp_c[:, i]
            t_idx = n_t

        return c_matrix

    def _set_compartment_order(self):
        compartment_order = [c for cmat in self.c_matrices
                             for c in cmat.compartment_order()]

        self.compartment_order = list(set(compartment_order))

    def shape(self):
        dim0 = sum([mat.shape()[0] for mat in self.c_matrices])
        dim1 = len(self.compartment_order)
        return (dim0, dim1)
