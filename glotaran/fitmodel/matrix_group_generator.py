""" Glotaran Fitmodel Matrix Group Generator"""

from typing import List, Generator, Tuple, Type
from collections import OrderedDict
import numpy as np

from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.model.parameter_group import ParameterGroup

from .matrix import Matrix
from .matrix_group import MatrixGroup


class MatrixGroupGenerator:
    """A MatrixGroupGenerator groups a set of matrices along the estimated or
    calculated axis.
    """
    def __init__(self, matrix: Type[Matrix], calculated=False):
        """

        Parameters
        ----------
        matrix : Type[Matrix]
            An implementation of fitmodel.Matrix
        calculated :
            (Default value = False)
            If true, group along the calculated insted of eastimated axis

        """
        self._groups = OrderedDict()
        self._matrix = matrix
        self._calculated = calculated

    @classmethod
    def for_model(cls, model: 'glotaran.Model', matrix: Type[Matrix], xtol=0.5, calculated=False):
        """ Creates matrix group generator for a full model.

        Parameters
        ----------
        model : glotaran.Model
            A Glotaran Model
        matrix : Type[fitmodel.Matrix]
            An implementation of fitmodel.Matrix
        xtol :
            (Default value = 0.5)
            Grouping tolerance
        calculated :
            (Default value = False)
            If true, group along the calculated insted of eastimated axis

        """
        gen = cls(matrix, calculated)
        gen.init_groups_for_model(model, xtol)
        return gen

    @classmethod
    def for_dataset(cls,
                    model: 'glotaran.Model',
                    dataset: str,
                    matrix: Type[Matrix],
                    calculated=False):
        """ Creates matrix group generator for a dataset.

        Parameters
        ----------
        model : glotaran.Model
            A Glotaran Model

        dataset: str
            Label of the dataset

        matrix : Type[fitmodel.Matrix]
            An implementation of fitmodel.Matrix

        calculated :
             (Default value = False)
            If true, group along the calculated insted of eastimated axis

        """
        gen = cls(matrix, calculated)
        data = model.datasets[dataset]
        gen.add_dataset_to_group(model, data, 0)
        return gen

    def init_groups_for_model(self, model: 'glotaran.Model', xtol: float):
        """ Initializes the matrix groups for a model.

        Parameters
        ----------
        model : glotaran.Model
            A Glotaran Model

        xtol :
            Grouping tolerance

        """
        for _, dataset in model.datasets.items():
            self.add_dataset_to_group(model, dataset, xtol)

    def add_dataset_to_group(self,
                             model: 'glotaran.Model',
                             dataset: DatasetDescriptor,
                             xtol: float):
        """ Adds a dataset to the group

        Parameters
        ----------
        model : glotaran.Model
            A Glotaran Model

        dataset: glotaran.DatasetDescriptor
            Dataset descriptor

        xtol :
            Grouping tolerance

        """
        grouping_axis = dataset.dataset.get_calculated_axis() if self._calculated\
            else dataset.dataset.get_estimated_axis()
        for matrix in [self._matrix(x, dataset, model) for x
                       in grouping_axis]:
            self._add_matrix_to_group(matrix, xtol)

    def _add_matrix_to_group(self, matrix: Matrix, xtol: float):
        if matrix.index in self._groups:
            self._groups[matrix.index].add_matrix(matrix)
        elif any(abs(matrix.index-val) < xtol for val in self._groups):
            idx = [val for val in self._groups if abs(matrix.index-val) <
                   xtol][0]
            self._groups[idx].add_matrix(matrix)
        else:
            self._groups[matrix.index] = MatrixGroup(matrix)

    def groups(self) -> Generator[MatrixGroup, None, None]:
        """Generator returning all groups."""
        for _, group in self._groups.items():
            yield group

    def groups_in_range(self, rng: Tuple[float, float]) -> List[MatrixGroup]:
        """Returns a list of all groups within a range.

        Parameters
        ----------
        rng : tuple(float, float)
            Range


        Returns
        -------
        groups : list(fitmodel.MatrixGroup)
            All groups in range.


        """
        return [g for g in self.groups() if rng[0] <= g.x <= rng[1]]

    def calculate(self, parameter: ParameterGroup) -> List[np.array]:
        """ Calculates the matrices in the group.

        Parameters
        ----------
        parameter : glotaran.ParameterGroup

        Returns
        -------
        matrices : list(numpy.array)

        """
        return [group.calculate(parameter) for group in self.groups()]

    def create_dataset_group(self) -> List[np.array]:
        """ Groups all datasets in the model and concats them on the calculated
        axis

        Returns
        -------
        datasets : list(numpy.array)
            A list with the grouped datasets
        """

        dataset_group = []
        for _, group in self._groups.items():
            data = np.array([])
            for mat in group.matrices:
                index = np.where(mat.dataset.dataset.get_estimated_axis() ==
                                 mat.index)
                data = np.concatenate((data,
                                       mat.dataset.dataset.get()[index, :].flatten()))
            dataset_group.append(data)
        return dataset_group
