""" Glotaran K-Matrix """

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import scipy

from glotaran.model import model_item

from .initial_concentration import InitialConcentration


@model_item(
    attributes={
        'matrix': {'type': Dict[Tuple[str, str], str], 'target': ('compartment', 'parameter')},
    },
)
class KMatrix:
    """ A K-Matrix represents a first order differental system."""

    @classmethod
    def empty(cls, label: str, compartments: List[str]):
        """ Creates an empty K-Matrix. Useful for combining.

        Parameters
        ----------
        label : str
            Label of the K-Matrix

        matrix : OrderedDict(tuple(str, str), str)
            Dictonary with the matrix entries in 'to-from' notation.

        compartments : list(str)
            A list of all compartments in the model.
        """
        return cls(label, OrderedDict())

    def involved_compartments(self) -> List[str]:
        """ A list of all compartments in the matrix. """
        compartments = []
        for index in self.matrix:
            compartments.append(index[0])
            compartments.append(index[1])

        compartments = list(set(compartments))
        return compartments

    def combine(self, k_matrix: "KMatrix") -> "KMatrix":
        """ Creates a combined matrix.

        Parameters
        ----------
        k_matrix : KMatrix
            Matrix to combine with.


        Returns
        -------
        combined : KMatrix
            The combined KMatrix.

        """
        #  if isinstance(k_matrix, list):
        #      next = k_matrix[1:] if len(k_matrix) > 2 else k_matrix[1]
        #      return self.combine(k_matrix[0]).combine(next)
        if not isinstance(k_matrix, KMatrix):
            raise TypeError("K-Matrices can only be combined with other"
                            "K-Matrices.")
        combined_matrix = OrderedDict()
        for entry in k_matrix.matrix:
            combined_matrix[entry] = k_matrix.matrix[entry]
        for entry in self.matrix:
            combined_matrix[entry] = self.matrix[entry]
        return KMatrix("{}+{}".format(self.label, k_matrix.label),
                       combined_matrix, self._all_compartments)

    def asarray(self, compartments: List[str]) -> np.ndarray:
        """ Depricated, only used for testing"""

        compartments = [c for c in compartments if c in self.involved_compartments()]
        size = len(compartments)
        array = np.zeros((size, size), dtype=np.float64)
        # Matrix is a dict
        for index in self.matrix:
            i = compartments.index(index[0])
            j = compartments.index(index[1])
            array[i, j] = self.matrix[index]
        return array

    def full(self, compartments: List[str]):
        """[ 0 k3
          k1 k2]

        translates to

        [ -k1 k3
          k1 -k2-k3]

        this still correct

        d/dt S1 = -k1 S1 + k3 S2
        d/dt S2 = +k1 S1 - k2 S2 - k3 S2

        it helps to read it so:


        [-k1     k3       [S1]
        k1  -k2-k3]    [S2]

        Parameters
        ----------
        parameter :


        Returns
        -------

        """
        compartments = [c for c in compartments if c in self.involved_compartments()]
        size = len(compartments)
        mat = np.zeros((size, size), np.float64)
        for (to_comp, from_comp), param in self.matrix.items():
            to_idx = compartments.index(to_comp)
            fr_idx = compartments.index(from_comp)

            if to_idx == fr_idx:
                mat[to_idx, fr_idx] -= param
            else:
                mat[to_idx, fr_idx] += param
                mat[fr_idx, fr_idx] -= param
        return mat

    def eigen(self, compartments: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the eigenvalues and eigenvectors of the k matrix.

        Parameters
        ----------
        parameter : glotaran.model.ParameterGroup


        Returns
        -------
        (eigenvalues, eigenvectos) : tuple(np.ndarray, np.ndarray)

        """
        # We take the transpose to be consistent with timp
        matrix = self.full(compartments).T
        # get the eigenvectors and values, we take the left ones to have
        # computation consistent with TIMP
        eigenvalues, eigenvectors = scipy.linalg.eig(matrix, left=True,
                                                     right=False)
        return (eigenvalues.real, eigenvectors.real)

    def _gamma(self,
               compartments,
               eigenvectors,
               initial_concentration: InitialConcentration) -> np.ndarray:
        k_compartments = [c for c in compartments if c in self.involved_compartments()]
        initial_concentration = \
            [initial_concentration.parameters[compartments.index(c)]
             for c in k_compartments]
        eigenvectors = scipy.linalg.inv(eigenvectors)
        gamma = np.matmul(eigenvectors, initial_concentration)

        return gamma

    def a_matrix(self,
                 compartments: List[str],
                 initial_concentration: InitialConcentration) -> np.ndarray:
        eigenvalues, eigenvectors = self.eigen(compartments)
        gamma = self._gamma(compartments, eigenvectors, initial_concentration)

        a_matrix = np.empty(eigenvectors.shape, dtype=np.float64)

        for i in range(eigenvectors.shape[0]):
            a_matrix[i, :] = eigenvectors[:, i] * gamma[i]

        return a_matrix
