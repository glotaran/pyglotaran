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
        #  print("GGGGGGG")
        #  print(parameter)
        #  print(data.shape)
        #  print(c_matrix.shape)
        res = np.empty(data.shape, dtype=np.float64)
        #  print(np.asarray([data[:, 0]]).T.shape)
        #  print(res.shape)
        #  print(res.shape)
        #  print(res.shape)
        #  print(res.flatten().shape)
        
        #  for i in range(data.shape[1]):
        #  print(data.shape[1])
        #  print(len(data[0,:]))
        for i in range(100):
        
            #  print(i)
            #  print(data[:,i].shape)
        #      b = data[:,i]
            #  qr = qr_decomposition(c_matrix, data[:, i])
            qr = qr_decomposition(c_matrix, np.asarray([data[:,i]]).T)[:,0]
            #  print(qr.shape)
            res[:, i] = qr
            #  res[:, i] = qr_decomposition(c_matrix, data[:,i])
            #  print("TTTTTTT")
            #  print(b.shape)
            #  print(qr.shape)
            #  print(c_matrix.shape)
            #  res[:, i] = qr            #  print(len(qr.shape))
            #  print(res[:,i].shape)
            #  res[:,i] = qr

        print("DDD")
        print(res[:10, 499])
        print(res[:10, 500])
        print(np.asarray([data[:,0]]).T.shape)
        #  res = qr_decomposition(c_matrix, data)
        return res.flatten()
