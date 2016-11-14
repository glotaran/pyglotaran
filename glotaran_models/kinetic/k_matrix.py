from scipy.sparse import dok_matrix
import numpy as np


class KMatrix(object):
    """
    A KMatrix has an label and a scipy.sparse matrix
    """
    def __init__(self, label, matrix):
        self.label = label
        self.matrix = matrix

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if not isinstance(value, dict):
            raise TypeError("Matrix must be dict like {(1,2):value}")
        self._matrix = value

    @property
    def compartment_map(self):
        compartments = []
        for index in self.matrix:
            compartments.append(index[0])
            compartments.append(index[1])

        compartments = list(set(compartments))
        return compartments

    def combine(self, k_matrix):
        if not isinstance(k_matrix, KMatrix):
            raise TypeError("K-Matrices can oly be combined with other"
                            "K-Matrices.")
        combined_matrix = {}
        for entry in k_matrix.matrix:
            combined_matrix[entry] = k_matrix.matrix[entry]
        for entry in self.matrix:
            combined_matrix[entry] = self.matrix[entry]
        return KMatrix("{}+{}".format(self.label, k_matrix.label),
                       combined_matrix)

    def asarray(self):
        compartment_map = self.compartment_map
        size = len(compartment_map)
        array = np.zeros((size, size), dtype=np.int32)
        for index in self.matrix:
            i = compartment_map.index(index[0])
            j = compartment_map.index(index[1])
            array[i, j] = self.matrix[index]
        return array

    def __str__(self):
        return "Label: {}\nMatrix:\n{}".format(self.label,
                                               self.asarray())
