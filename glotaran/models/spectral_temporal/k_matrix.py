""" Glotaran K-Matrix """

from collections import OrderedDict
import itertools
from typing import Dict, List, Tuple
import numpy as np
import scipy

from glotaran.model import model_item

from .initial_concentration import InitialConcentration


@model_item(
    attributes={
        'matrix': {'type': Dict[Tuple[str, str], str], 'target': (None, 'parameter')},
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

    def reduced(self, compartments: List[str]) -> np.ndarray:
        """ """

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
        (eigenvalues, eigenvectors) : tuple(np.ndarray, np.ndarray)

        """
        # We take the transpose to be consistent with timp
        matrix = self.full(compartments).T
        # get the eigenvectors and values, we take the left ones to have
        # computation consistent with TIMP
        eigenvalues, eigenvectors = scipy.linalg.eig(matrix, left=True,
                                                     right=False)
        return (eigenvalues.real, eigenvectors.real)

    def rates(self, compartments):
        if self.is_unibranched(compartments):
            return np.diag(self.full(compartments)).copy()
        else:
            rates, _ = self.eigen(compartments)
            return rates

    def _gamma(self,
               eigenvectors,
               initial_concentration: InitialConcentration) -> np.ndarray:
        compartments = [c for c in initial_concentration.compartments
                        if c in self.involved_compartments()]
        k_compartments = [c for c in compartments if c in self.involved_compartments()]
        initial_concentration = \
            [initial_concentration.parameters[compartments.index(c)]
             for c in k_compartments]
        eigenvectors = scipy.linalg.inv(eigenvectors)
        gamma = np.matmul(eigenvectors, initial_concentration)

        return gamma

    def a_matrix(self, initial_concentration: InitialConcentration) -> np.ndarray:
        if self.is_unibranched(initial_concentration.compartments):
            a_matrix = self.a_matrix_unibranch(initial_concentration)
        else:
            a_matrix = self.a_matrix_non_unibranch(initial_concentration)
        return a_matrix

    def a_matrix_non_unibranch(self, initial_concentration: InitialConcentration) -> np.ndarray:
        eigenvalues, eigenvectors = self.eigen(initial_concentration.compartments)
        gamma = self._gamma(eigenvectors, initial_concentration)

        a_matrix = np.empty(eigenvectors.shape, dtype=np.float64)

        for i in range(eigenvectors.shape[0]):
            a_matrix[i, :] = eigenvectors[:, i] * gamma[i]

        return a_matrix

    def a_matrix_unibranch(self, initial_concentration: InitialConcentration) -> np.array:
        compartments = [c for c in initial_concentration.compartments
                        if c in self.involved_compartments()]
        matrix = self.full(compartments).T
        rates = np.diag(matrix)

        a_matrix = np.zeros(matrix.shape, dtype=np.float64)
        a_matrix[0, 0] = 1.0
        for i, j in itertools.product(range(rates.size), range(1, rates.size)):
            if i > j:
                continue
            a_matrix[i, j] = np.prod([rates[m] for m in range(j)]) / \
                np.prod([rates[m] - rates[i] for m in range(j+1) if not i == m])
        return a_matrix

    def is_unibranched(self, compartments):
        matrix = self.reduced(compartments)
        for i in range(matrix.shape[1]):
            if not np.nonzero(matrix[:, i])[0].size == 1:
                return False
        return True
