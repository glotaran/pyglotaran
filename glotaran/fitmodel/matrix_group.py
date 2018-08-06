"""Glotaran Fitmodel Matrix Group"""

from typing import List, Tuple
import collections
import numpy as np

from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.model.parameter_group import ParameterGroup

from .matrix import Matrix


class MatrixGroup:
    """A group of matrices at an index on the estimated or calculated axis. The
    matrices are concatinated along estimated or calculated axis.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, matrix: Matrix):
        """

        Parameters
        ----------
        matrix: Matrix
            The initial matrix

        """
        self.matrices = [matrix]

    def add_matrix(self, matrix: Matrix):
        """Adds matrix to the group.

        Parameters
        ----------
        matrix : fitmodel.Matrix
            Matrix to be added.

        """
        self.matrices.append(matrix)

    def calculate(self, parameter: ParameterGroup) -> np.array:
        """Calculates the matrix group.

        Parameters
        ----------
        parameter : ParameterGroup
            Parameters for the calculation

        Returns
        -------
        matrix : numpy.array
            Calculated matrix


        """

        compartment_order = self.compartment_order()

        matrix = np.zeros(self.shape(), dtype=np.float64)

        t_idx = 0

        for mat in self.matrices:
            n_t = t_idx + mat.shape[0]
            mat.calculate(matrix[t_idx:n_t, :], compartment_order,
                          parameter)
            mat.apply_constraints(matrix[t_idx:n_t, :], compartment_order,
                                  parameter)
            t_idx = n_t

        return matrix

    def matrix_location(self, dataset: DatasetDescriptor) -> Tuple[int, int]:
        """Returns the location of a dataset's matrix within the matrix group

        Parameters
        ----------
        dataset : DatasetDescriptor
            Descriptor of the dataset

        Returns
        -------
        location : tuple(int, int)
            The location of the datset. None if dataset is not present.

        """
        t_idx = 0
        for mat in self.matrices:
            if mat.dataset.label == dataset.label:
                return (t_idx, mat.shape()[0])
            t_idx += mat.shape()[0]
        return None

    def sizes(self) -> List[int]:
        """Returns list with all sizes of the matrices in the concating
        direction.

        Returns
        -------
        sizes : list(int)

        """
        return [mat.shape[0] for mat in self.matrices]

    def compartment_order(self) -> List[str]:
        """A list of compartments which maps entries in the matrix to
        compartments in the model.

        Returns
        -------
        compartment_order : list(str)

        """
        compartment_order = [c for cmat in self.matrices
                             for c in cmat.compartment_order]
        return list(collections.OrderedDict.fromkeys(compartment_order).keys())

    def shape(self) -> Tuple[int, int]:
        """The shape of the whole matrix group.

        Returns
        -------
        shape : tuple(int, int)
        """
        dim0 = sum([mat.shape[0] for mat in self.matrices])
        dim1 = len(self.compartment_order())
        return (dim0, dim1)
