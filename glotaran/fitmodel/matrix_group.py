import numpy as np


class MatrixGroup(object):
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
