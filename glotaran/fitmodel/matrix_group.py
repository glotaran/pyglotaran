import numpy as np


class MatrixGroup(object):
    def __init__(self, c_matrix):
        self.id = c_matrix.x
        self.c_matrices = [c_matrix]

    def add_cmatrix(self, c_matrix):
        self.c_matrices.append(c_matrix)

    def calculate(self, parameter):

        compartment_order = self.compartment_order()

        c_matrix = np.zeros(self.shape(), dtype=np.float64)

        t_idx = 0

        for cmat in self.c_matrices:
            n_t = t_idx + cmat.shape()[0]
            cmat.calculate(c_matrix[t_idx:n_t, :], compartment_order,
                           parameter)
            t_idx = n_t

        return c_matrix

    def compartment_order(self):
        compartment_order = [c for cmat in self.c_matrices
                             for c in cmat.compartment_order()]
        return list(set(compartment_order))

    def shape(self):
        dim0 = sum([mat.shape()[0] for mat in self.c_matrices])
        dim1 = len(self.compartment_order())
        return (dim0, dim1)
