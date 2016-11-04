from lmfit import Minimizer
import numpy as np

from .qr_decomposition import qr_decomposition


class VariableProjectionMinimizer(Minimizer):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(VariableProjectionMinimizer, self).__init__(self.uuu,
                                                          model.parameter(),
                                                          fcn_args=args,
                                                          fcn_kws=kwargs)

    def uuu(self, parameter, *args, **kwargs):
        print(parameter)
        data = kwargs['data']
        print("iiiiiii")
        c_matrix = self.model.c_matrix(parameter.valuesdict(), *args, **kwargs)
        print(c_matrix.shape)
        print(data.shape)
        print("jjjjjjj")
        res = np.empty(data.shape, dtype=np.float64)
        #  C = calculateC(k, times)
        for i in range(data.shape[1]):
            b = data[:, i]
            res[:, i] = qr_decomposition(c_matrix, b)
        #  print(res.flatten())

        print(np.sum(res))
        return res.flatten()
