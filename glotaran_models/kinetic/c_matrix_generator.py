from .model import KineticModel
from glotaran_core.fitting.variable_projection import CMatrixGenerator


class KineticCMatrixGenerator(CMatrixGenerator):
    def __init__(self, model):
        self._model = model

    def get_nr_of_compartements(self):
        return self._model.k_matrices[0].shape[0]

    def get_kinetic_parameter(self):
        kinetic_parameter = []
        for i in range(self.get_nr_of_compartements):
            par_index = self._model.k_matrices[0][i, i]
            par_val = self._model.parameter[par_index].value
            kinetic_parameter.append(par_val)
        return kinetic_parameter
