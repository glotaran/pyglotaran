from lmfit_varpro import SeparableModel

from .matrix_group_generator import MatrixGroupGenerator
from .result import Result


class FitModel(SeparableModel):

    def __init__(self, model):
        self._model = model
        self._generator = None
        self._dataset_group = None

    def get_initial_fitting_parameter(self):
        return self._model.parameter.as_parameters_dict()

    def data(self, **kwargs):
        return self._dataset_group

    def fit(self, *args, **kwargs):
        self._generator = MatrixGroupGenerator.for_model(self._model,
                                                         self._model.
                                                         calculated_matrix())
        self._dataset_group = self._generator.create_dataset_group()
        result = Result(self, self.get_initial_fitting_parameter(),
                        *args, **kwargs)
        result.fit(*args, **kwargs)
        return result

    def c_matrix(self, parameter, **kwargs):
        parameter = parameter.valuesdict()
        if "dataset" in kwargs:
            label = kwargs["dataset"]
            gen = MatrixGroupGenerator.for_dataset(self._model, label,
                                                   self._model.
                                                   calculated_matrix())
        else:
            gen = self._generator
            if gen is None:
                gen = MatrixGroupGenerator.for_model(self._model,
                                                     self._model.
                                                     calculated_matrix())
        return gen.calculate(parameter)

    def e_matrix(self, parameter, **kwargs):
        parameter = parameter.valuesdict()
        dataset = kwargs["dataset"]
        gen = MatrixGroupGenerator.for_dataset(self._model, dataset,
                                               self._model.estimated_matrix(),
                                               calculated=True)
        return gen.calculate(parameter)


def isclass(obj, classname):
    return obj.__class__.__name__ == classname
