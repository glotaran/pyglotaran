import numpy as np


class CMatrix(object):

    def __init__(self, matrix, compartment_map, independent_axies):
        if not isinstance(matrix, np.ndarray):
            raise TypeError(" Matrix must be ndarray.")
        self.matrix = matrix
        self.compartment_map = compartment_map
        self.independent_axies = independent_axies

    def combine_megacomplex_matrices(self, c_matrix):
        smaller, larger = [None]*2
        if self.nr_compartments() > c_matrix.nr_compartments:
            smaller = c_matrix
            larger = self
        else:
            smaller = self
            larger = c_matrix

        combined = np.copy(larger.matrix)

        for i in range(smaller.nr_compartments()):
            compartment = smaller.compartment_map[i]
            j = larger.compartment_map.index(compartment)
            combined[:, :, j] = combined[:, :, j] + smaller.matrix[i]

        return CMatrix(combined, larger.compartment_map)

    def combine_dataset_matrices(self, c_matrix, tolerance=0.1):

        smaller, larger = [None]*2

        if self.independent_axies.shape[0] > c_matrix.independent_axies[0]:
            smaller = c_matrix
            larger = self
        else:
            smaller = self
            larger = c_matrix

        smaller_axies = smaller.independent_axies
        larger_axies = larger.independent_axies

        non_overlapping = [val for val in smaller.independent_axies if not
                           any(abs(x-val) < tolerance for x in
                               larger.independent_axies)]

        combined_axies = larger.independent_axies + non_overlapping

        size_x1_axies = len(combined_axies)
        size_x2_axies = c_matrix.matrix.shape[1] + self.matrix.shape[0]
        size_compartments = max([self.nr_compartments(),
                                 c_matrix.nr_compartments()])

        # F841 local variable 'combined' is assigned to but never used
        combined = np.empty((size_x1_axies, size_x2_axies, size_compartments), dtype=np.float64)

        # F841 local variable 'smaller_axis_val_tmp' is assigned to but never used
        # F841 local variable 'tmp' is assigned to but never used
        for i in range(size_x1_axies):
            if i in range(larger.independent_axies):
                smaller_axis_val_tmp = [val for val in
                                        smaller_axies if abs(val -
                                                             larger_axies[i]) <
                                        tolerance]
                tmp = smaller

    def nr_compartments(self):
        return self.matrix.shape[2]
