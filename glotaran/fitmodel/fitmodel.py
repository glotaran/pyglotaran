import numpy as np
from lmfit_varpro import CompartmentEqualityConstraint, SeparableModel

from .matrix_group_generator import MatrixGroupGenerator
from .result import Result


class FitModel(SeparableModel):

    def __init__(self, model):
        self._model = model
        self._generator = None
        self._dataset_group = None

    def get_initial_fitting_parameter(self):
        return self._model.parameter.as_parameters_dict(only_fit=True)

    def data(self, **kwargs):
        return self._dataset_group

    def fit(self, nnls=False, *args, **kwargs):

        result = self.result(nnls, *args, **kwargs)

        result.fit(*args, **kwargs)
        return result

    def result(self, nnls, *args, **kwargs):
        self._generator = MatrixGroupGenerator.for_model(self._model,
                                                         self._model.
                                                         calculated_matrix())
        self._dataset_group = self._generator.create_dataset_group()
        c_constraints = []
        for _, dataset in self._model.datasets.items():
            constraints = [c for c in dataset.compartment_constraints if
                           c.type() == 2]

            for c in constraints:
                for interval in c.intervals:
                    g = list(self._generator.groups_in_range(interval))[0]
                    crange = g.get_dataset_location(dataset)
                    i = g.compartment_order.index(c.target)
                    j = g.compartment_order.index(c.compartment)
                    c_constraints.append(
                        CompartmentEqualityConstraint(c.weight, i, j, interval,
                                                      crange))
        result = Result(self, self.get_initial_fitting_parameter(),
                        equality_constraints=c_constraints, nnls=nnls,
                        *args, **kwargs)
        return result

    def c_matrix(self, parameter, *args, **kwargs):
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

    def e_matrix(self, parameter, *args, **kwargs):

        # We don't have a way to construct a complete E matrix for the full
        # problem yet.
        if "dataset" not in kwargs:
            raise Exception("'dataset' non specified in kwargs")

        parameter = parameter.valuesdict()
        dataset = self._model.datasets[kwargs["dataset"]]

        # A data object needs to be present to provide axies
        if dataset.data is None:
            raise Exception("No Data object present for dataset '{}'"
                            .format(kwargs["dataset"]))

        x = kwargs["x"] if "x" in kwargs else np.asarray([0])

        e_matrix = self._model.estimated_matrix()(x, dataset, self._model)
        return e_matrix.calculate_standalone(parameter)


def isclass(obj, classname):
    return obj.__class__.__name__ == classname
