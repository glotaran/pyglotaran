from lmfit import Model as LmFitModel
from glotaran_core.model import Model

class FitModel(LmFitModel):
    def __init__(self, model):
        self.model = model
        super(FitModel, self).__init__()

    @property
    def model(self):
        return self.model

    @model.setter
    def model(self, value):
        if not isinstance(value, Model):
            raise TypeError
        self._model = value

    def _func(self, t, params):
    res = np.empty(PSI.shape, dtype=np.float64)
    C = self._calculate_c_matrix(k, times)
    for i in range(PSI.shape[1]):
        b = PSI[:,i]
        res[:,i] = qr(C, b)

    return res.flatten()

    def _calculate_c_matrix(self):
        return self.model.calculate_c_matrix()
