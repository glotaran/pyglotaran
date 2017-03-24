from collections import OrderedDict
import numpy as np


class KMatrix(object):
    """
    A KMatrix has an label and a scipy.sparse matrix
    """
    def __init__(self, label, matrix, compartments):
        self.label = label
        self.matrix = matrix
        self._create_compartment_map(compartments)

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
        if not isinstance(value, OrderedDict):
            raise TypeError("Matrix must be OrderedDict like {(1,2):value}")
        self._matrix = value

    @property
    def involved_compartments(self):
        compartments = []
        for index in self.matrix:
            compartments.append(index[0])
            compartments.append(index[1])

        compartments = list(set(compartments))
        return compartments

    @property
    def compartment_map(self):
        return self._compartment_map

    def _create_compartment_map(self, compartments):
        self._compartment_map = [c for c in compartments if c in
                                 self.involved_compartments]

    def combine(self, k_matrix):
        if isinstance(k_matrix, list):
            next = k_matrix[1:] if len(k_matrix) > 2 else k_matrix[1]
            return self.combine(k_matrix[0]).combine(next)
        if not isinstance(k_matrix, KMatrix):
            raise TypeError("K-Matrices can only be combined with other"
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
        # Matrix is a dict
        for index in self.matrix:
            i = compartment_map.index(index[0])
            j = compartment_map.index(index[1])
            array[i, j] = self.matrix[index]
        return array

    def __str__(self):
        return "Label: {}\nMatrix:\n{}".format(self.label,
                                               self.asarray())
