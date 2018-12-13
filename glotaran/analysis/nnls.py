import numpy as np
from scipy.optimize import nnls


def residual_nnls(matrix, data):

    clp, _ = nnls(matrix, data)
    residual = np.dot(matrix, clp) - data
    return clp, residual
