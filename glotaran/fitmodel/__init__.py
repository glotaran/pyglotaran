import numpy as np

from . import (c_matrix, matrix_group_generator, fitmodel)

FitModel = fitmodel.FitModel
MatrixGroupGenerator = matrix_group_generator.MatrixGroupGenerator
CMatrix = c_matrix.CMatrix


def parameter_map(parameter):
    def map_fun(i):
        if i != 0:
            i = parameter["p_{}".format(int(i))]
        return i
    return np.vectorize(map_fun)


def parameter_idx_to_val(parameter, index):
        if index != 0:
            index = parameter["p_{}".format(int(index))]
        return index
