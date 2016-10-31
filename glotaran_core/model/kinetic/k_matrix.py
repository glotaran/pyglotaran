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
        if not isinstance(value, str):
            raise TypeError("Labels must be strings.")
        self._label = value

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if not isinstance(value, dict):
            raise TypeError("Matrix must be sparse dict like {(1,2):value}")
        size = 0
        for index in value:
            s = max(index)
            if s > size:
                size = s
        m = dok_matrix((size, size), dtype=np.int32)
        for index in value:
            m[index[0]-1, index[1]-1] = value[index]
        self._matrix = m


    def __str__(self):
        return "Label: {}\nMatrix:\n{}".format(self.label(),
                                               self.matrix().toarray())
