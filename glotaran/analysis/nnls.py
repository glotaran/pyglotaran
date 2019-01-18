import numpy as np
from scipy.optimize import nnls


def residual_nnls(matrix, data):

    clp, _ = nnls(matrix, data)
    residual = data - np.dot(matrix, clp)
    return clp, residual
