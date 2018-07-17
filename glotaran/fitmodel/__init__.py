"""Glotarans fitmodel package"""
import numpy as np

from . import (matrix, matrix_group_generator, fitmodel, result)

FitModel = fitmodel.FitModel
MatrixGroupGenerator = matrix_group_generator.MatrixGroupGenerator
Matrix = matrix.Matrix
Result = result.Result


def parameter_map(parameter):
    def map_fun(i):
        return parameter_idx_to_val(parameter, i)
    return np.vectorize(map_fun)


def parameter_idx_to_val(parameter, index):
    if isinstance(index, (float, str)):
        try:
            index = int(index)
            if index == 0:
                return index
        except:
            index = index.replace('.', '_')
    return parameter["p_{}".format(index)]
