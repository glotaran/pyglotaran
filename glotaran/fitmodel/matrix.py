"""Glotaran Fitmodel Matrix"""

from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np

from glotaran.model.compartment_constraints import CompartmentConstraintType
from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.model.parameter_group import ParameterGroup


class Matrix(ABC):
    """An abstract matrix which gets prepared with a dataset and a model and
    can be calculated by giving a ParameterGroup."""

    def __init__(self, index: any, dataset: DatasetDescriptor, model:
                 'glotaran.Model'):
        """

        Parameters
        ----------
        x: any
            Index in the matrix group.

        dataset : glotaran.DatasetDescriptor
            Dataset descriptor for the matrix.

        model: glotaran.Model
            Model for the matrix.


        """
        self.index = index
        self._dataset = dataset
        self._model = model

    def calculate_standalone(self, parameter: ParameterGroup) -> np.array:
        """ Calculates the matrix outside a matrix group.

        Parameters
        ----------
        parameter : glotaran.ParameterGroup
            Parameters to calculate the matrix with.


        Returns
        -------
        matrix : numpy.array
            The calculated matrix.

        """
        matrix = np.zeros((self.shape), np.float64)
        self.calculate(matrix, self.compartment_order, parameter)
        self.apply_constraints(matrix, self.compartment_order, parameter)
        return matrix

    def apply_constraints(self,
                          matrix: np.ndarray,
                          compartment_order: List[str],
                          parameter: ParameterGroup):
        """Applies Zero and Equal constraints to the matrix.

        Parameters
        ----------
        matrix: np.array
            Target matrix.

        compartment_order: List[str]
            A list of compartments representing the  mapping of compartments
            to entries in the target matrix.

        parameter: ParameterGroup
            Parameters to calculate the matrix with.

        """
        # get constraint compartment indeces
        constraints = [c for c in self.dataset.compartment_constraints
                       if c.applies(self.index)
                       and c.type() is not CompartmentConstraintType.equal_area]
        for constrain in constraints:
            idx = compartment_order.index(constrain.compartment)

            matrix[:, idx].fill(0.0)
            if constrain.type() == CompartmentConstraintType.equal:
                for target, param in constrain.targets_and_parameter:
                    t_idx = compartment_order.index(target)
                    param = parameter.get(param)
                    matrix[:, idx] += param * matrix[:, t_idx]

    @property
    def dataset(self) -> DatasetDescriptor:
        """The underlying glotaran.DatasetDescriptor"""
        return self._dataset

    @property
    def model(self) -> 'glotaran.Model':
        """The underlying glotaran.Model"""
        return self._model

    @property
    def compartment_order(self) -> List[str]:
        """A list of compartments representing the internal mapping of
        compartments to entries in the matrix."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, int]:
        """A tuple representing the dimensions. First entry is the compartment
        index. Use Matrix.compartment_order to map entries to compartments."""
        raise NotImplementedError

    @abstractmethod
    def calculate(self, matrix: np.array, compartment_order: List[str], parameter: ParameterGroup):
        """Calculates the matrix.

        Parameters
        ----------
        matrix: np.array
            Target matrix.

        compartment_order: List[str]
            A list of compartments representing the  mapping of compartments
            to entries in the target matrix.

        parameter: ParameterGroup
            Parameters to calculate the matrix with.

        """
        raise NotImplementedError
