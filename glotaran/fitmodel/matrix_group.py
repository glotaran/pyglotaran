import numpy as np


class MatrixGroup(object):
    def __init__(self, matrix):
        self.x = matrix.x
        self.matrices = [matrix]

    def add_matrix(self, c_matrix):
        self.matrices.append(c_matrix)

    def calculate(self, parameter):

        compartment_order = self.compartment_order()

        matrix = np.zeros(self.shape(), dtype=np.float64)

        t_idx = 0

        for mat in self.matrices:
            n_t = t_idx + mat.shape()[0]
            mat.calculate(matrix[t_idx:n_t, :], compartment_order,
                          parameter)
            t_idx = n_t

        return matrix

    def matrix_location(self, dataset):
        t_idx = 0
        for m in self.matrices:
            if m.dataset.label == dataset.label:
                return (t_idx, m.shape()[0])
            t_idx += m.shape()[0]

    def sizes(self):
        return [mat.shape()[0] for mat in self.matrices]

    def compartment_order(self):
        compartment_order = [c for cmat in self.matrices
                             for c in cmat.compartment_order()]
        return list(set(compartment_order))

    def shape(self):
        dim0 = sum([mat.shape()[0] for mat in self.matrices])
        dim1 = len(self.compartment_order())
        return (dim0, dim1)
