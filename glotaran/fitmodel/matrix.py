import numpy as np


class Matrix(object):
    def __init__(self, x, dataset, model):
        self.x = x
        self.dataset = dataset
        #  self.buffer = buffer

    def calculate_standalone(self, parameter):
        matrix = np.zeros(self.shape(), np.float64)
        self.calculate(matrix, self.compartment_order(), parameter)
        return matrix

    def calculate(self, matrix, compartment_order, parameter):
        raise NotImplementedError

    def compartment_order(self):
        raise NotImplementedError

    def shape(self):
        raise NotImplementedError
