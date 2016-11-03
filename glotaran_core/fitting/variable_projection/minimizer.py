from lmfit import Minimizer
import numpy as np


class VariableProjectionMinimizer(Minimizer):
    def __init__(self, c_matrix_generator):
        self.c_matrix_generator = c_matrix_generator

    def residual(parameter, *args, **kwargs):
        res = np.empty(PSI.shape, dtype=np.float64)
        C = calculateC(k, times)
        for i in range(PSI.shape[1]):
            b = PSI[:,i]
            res[:,i] = qr(C, b)

        return res.flatten()

