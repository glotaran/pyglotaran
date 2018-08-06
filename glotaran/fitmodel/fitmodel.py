"""Glotaran Fitmodel"""

from typing import List, Type
import numpy as np
from lmfit_varpro import CompartmentEqualityConstraint, SeparableModel
from lmfit import Parameters

from glotaran.model.parameter_group import ParameterGroup

from .matrix_group_generator import MatrixGroupGenerator
from .result import Result


class FitModel(SeparableModel):
    """FitModel is an implementation of lmfit-varpor.SeparableModel."""

    # pylint: disable=no-self-use
    # pylint: disable=arguments-differ
    # we do some convinince wrapping

    def __init__(self, model: 'glotaran.Model'):
        """

        Parameters
        ----------
        model : glotaran.Model

        """
        self._model = model
        self._generator = None
        self._dataset_group = None

    @property
    def model(self):
        """The underlying glotaran.Model"""
        return self._model

    def get_initial_fitting_parameter(self) -> Parameters:
        """

        Returns
        -------
        Parameters : lmfit.Parameters
        """
        return self._model.parameter.as_parameters_dict(only_fit=True)

    def data(self, **kwargs) -> List[np.array]:
        """ Returns the data to fit.


        Returns
        -------
        data: list(np.array)
        """
        if "dataset" in kwargs:
            label = kwargs["dataset"]
            gen = MatrixGroupGenerator.for_dataset(self._model, label,
                                                   self._model.
                                                   calculated_matrix())
            return gen.create_dataset_group()
        return self._dataset_group

    def fit(self, *args, nnls=False, **kwargs) -> Result:
        """Fits the model.

        Parameters
        ----------
        nnls :
             (Default value = False)
             Use Non-Linear Least Squares instead of variable projection.
        *args :

        **kwargs :


        Returns
        -------
        result : Result

        """

        result = self.result(nnls, *args, **kwargs)

        result.fit(*args, **kwargs)
        return result

    def result_class(self) -> Type[Result]:
        """Returns a Result class implementation. Meant to be overwritten."""
        return Result

    def result(self, nnls: bool, *args, **kwargs) -> Result:
        """Creates a Result object.

        Parameters
        ----------
        nnls : bool
             Use Non-Linear Least Squares instead of variable projection.

        *args :

        **kwargs :


        Returns
        -------
        result : Result
        """
        self._generator = MatrixGroupGenerator.for_model(self._model,
                                                         self._model.
                                                         calculated_matrix())
        self._dataset_group = self._generator.create_dataset_group()
        c_constraints = self._create_constraints()
        result = self.result_class()(self,
                                     self.get_initial_fitting_parameter(),
                                     nnls,
                                     c_constraints,
                                     *args, **kwargs)
        return result

    def c_matrix(self, parameter: Parameters, *args, **kwargs) -> np.array:
        """Implementation of SeparableModel.c_matrix.

        Parameters
        ----------
        parameter : lmfit.Parameters

        *args :

        **kwargs :
            dataset : str
                Only evaluate for the given dataset


        Returns
        -------
        matrix : np.array
        """
        parameter = ParameterGroup.from_parameter_dict(parameter)
        if "dataset" in kwargs:
            label = kwargs["dataset"]
            gen = MatrixGroupGenerator.for_dataset(self._model, label,
                                                   self._model.
                                                   calculated_matrix())
        else:
            if self._generator is None:
                self._init_generator()
            gen = self._generator
        return gen.calculate(parameter)

    def get_calculated_matrix_group(self, dataset=None):
        if dataset is None:
            return MatrixGroupGenerator.for_model(self._model,
                                                  self._model.
                                                  calculated_matrix())

        return MatrixGroupGenerator.for_dataset(self._model, dataset,
                                                self._model.
                                                calculated_matrix())

    def _init_generator(self):
        self._generator = MatrixGroupGenerator.for_model(self._model,
                                                         self._model.
                                                         calculated_matrix())

    def e_matrix(self, parameter, *args, **kwargs) -> np.array:
        """Implementation of SeparableModel.e_matrix.

        Parameters
        ----------
        parameter : lmfit.Parameters

        *args :

        **kwargs :
            dataset : str
                Only evaluate for the given dataset
            axis : np.array
                The axis to evaluate the e-matrix on.


        Returns
        -------
        matrix : np.array
        """
        # We don't have a way to construct a complete E matrix for the full
        # problem yet.
        if "dataset" not in kwargs:
            raise Exception("'dataset' non specified in kwargs")

        parameter = ParameterGroup.from_parameter_dict(parameter)
        dataset = self._model.datasets[kwargs["dataset"]]

        # A data object needs to be present to provide axies
        if dataset.dataset is None:
            raise Exception("No Data object present for dataset '{}'"
                            .format(kwargs["dataset"]))

        axis = kwargs["axis"] if "axis" in kwargs else np.asarray([0])

        e_matrix = self._model.estimated_matrix()(axis, dataset, self._model)
        return e_matrix.calculate_standalone(parameter)

    def _create_constraints(self) -> List[CompartmentEqualityConstraint]:
        c_constraints = []
        for _, dataset in self._model.datasets.items():
            constraints = [c for c in dataset.compartment_constraints if
                           c.type() == 2]

            for cons in constraints:
                for interval in cons.intervals:
                    group = list(self._generator.groups_in_range(interval))[0]
                    crange = group.get_dataset_location(dataset)
                    i = group.compartment_order.index(cons.target)
                    j = group.compartment_order.index(cons.compartment)
                    c_constraints.append(
                        CompartmentEqualityConstraint(cons.weight,
                                                      i, j,
                                                      cons.parameter,
                                                      interval,
                                                      crange))
        return c_constraints


def isclass(obj, classname):
    """ Checks if an objects classname matches the given classname

    Parameters
    ----------
    obj : any

    classname : str


    Returns
    -------
    isclass : bool
    """
    return obj.__class__.__name__ == classname
