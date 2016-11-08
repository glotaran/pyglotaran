from lmfit import Minimizer
import numpy as np

from .qr_decomposition import qr_decomposition


class VariableProjectionMinimizer(Minimizer):
    def __init__(self, model, initial_parameter, *args, **kwargs):
        self.model = model
        super(VariableProjectionMinimizer, self).__init__(self.residual,
                                                          initial_parameter,
                                                          fcn_args=args,
                                                          fcn_kws=kwargs)

    def residual(self, parameter, *args, **kwargs):
        data = kwargs['data']
        c_matrix = self.model.c_matrix(parameter.valuesdict(), *args, **kwargs)
        res = np.empty(data.shape, dtype=np.float64)
        for i in range(data.shape[1]):
        #  for i in range(1):

            b = data[:, i]
            qr = qr_decomposition(c_matrix, b)
            res[:, i] = qr

        return res.flatten()
