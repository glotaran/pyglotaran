"""Glotaran Fitmodel"""

from typing import List, Type
import numpy as np
from lmfit_varpro import CompartmentEqualityConstraint, SeparableModel
from lmfit import Parameters

from glotaran.model import ModelError, ParameterGroup

from .result import Result


from .grouping import create_group, calculate_group, get_data_group


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

        if not model.valid():
            raise ModelError(model)

        self._group = create_group(model)
        self._dataset_group = get_data_group(self._group)
        self._model = model

    @property
    def model(self):
        """The underlying glotaran.Model"""
        return self._model

    def data(self, **kwargs) -> List[np.ndarray]:
        """ Returns the data to fit.


        Returns
        -------
        data: list(np.ndarray)
        """
        if "dataset" in kwargs:
            dataset = kwargs['dataset']
            group = create_group(self._model, dataset=dataset)
            return get_data_group(group)
        return self._dataset_group

    def fit(self, parameter: ParameterGroup, *args, nnls=False, **kwargs) -> Result:
        """Fits the model.

        Parameters
        ----------
        parameter: ParameterGroup
        nnls :
             (Default value = False)
             Use Non-Linear Least Squares instead of variable projection.
        *args :

        **kwargs :


        Returns
        -------
        result : Result


        """

        result = self.result(parameter, nnls, *args, **kwargs)

        result.fit(*args, **kwargs)
        return result

    def result_class(self) -> Type[Result]:
        """Returns a Result class implementation. Meant to be overwritten."""
        return Result

    def result(self, parameter: ParameterGroup, nnls: bool, *args, **kwargs) -> Result:
        """Creates a Result object.

        Parameters
        ----------
        parameter: ParameterGroup
        nnls : bool
             Use Non-Linear Least Squares instead of variable projection.

        *args :

        **kwargs :


        Returns
        -------
        result : Result
        """
        c_constraints = self._create_constraints()
        result = self.result_class()(self,
                                     parameter,
                                     nnls,
                                     c_constraints,
                                     *args,
                                     nan_policy="omit",
                                     **kwargs,
                                     )
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
        if "dataset" in kwargs:
            dataset = kwargs['dataset']
            group = create_group(self._model, dataset=dataset)
            return calculate_group(group, self._model, parameter)
        return calculate_group(self._group, self._model, parameter)

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

        dataset = kwargs['dataset']
        group = create_group(self._model, group_axis='calculated', dataset=dataset)
        return calculate_group(group, self._model, parameter, matrix='estimated')

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
