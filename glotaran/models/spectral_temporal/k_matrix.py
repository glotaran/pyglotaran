""" Glotaran K-Matrix """

from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import scipy

from glotaran.model import InitialConcentration, ParameterGroup


class KMatrix:
    """ A K-Matrix represents a first order differental system."""
    def __init__(self, label: str, matrix: OrderedDict, compartments: List[str]):
        """

        Parameters
        ----------
        label : str
            Label of the K-Matrix

        matrix : OrderedDict(tuple(str, str), str)
            Dictonary with the matrix entries in 'to-from' notation.

        compartments : list(str)
            A list of all compartments in the model.

        """
        self.label = label
        self.matrix = matrix
        self._create_compartment_map(compartments)
        # We keep track of all compartments to combine later.
        self._all_compartments = compartments

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
        return cls(label, OrderedDict(), compartments)

    @property
    def label(self) -> str:
        """ Label of the K-Matrix"""
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def matrix(self) -> OrderedDict:
        """ Dictonary with the matrix entries in 'to-from' notation. """
        return self._matrix

    @matrix.setter
    def matrix(self, value: OrderedDict):
        if not isinstance(value, OrderedDict):
            raise TypeError("Matrix must be OrderedDict like {(1,2):value}")
        self._matrix = value

    @property
    def involved_compartments(self) -> List[str]:
        """ A list of all compartments in the matrix. """
        compartments = []
        for index in self.matrix:
            compartments.append(index[0])
            compartments.append(index[1])

        compartments = list(set(compartments))
        return compartments

    @property
    def compartment_map(self):
        """ A list of all compartments in the matrix. Index in the list
        correlates to index in the matrix."""
        return self._compartment_map

    def _create_compartment_map(self, compartments):
        self._compartment_map = [c for c in compartments if c in
                                 self.involved_compartments]

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

    def asarray(self) -> np.ndarray:
        """ Depricated, only used for testing"""
        compartment_map = self.compartment_map
        size = len(compartment_map)
        array = np.zeros((size, size), dtype=np.int32)
        # Matrix is a dict
        for index in self.matrix:
            i = compartment_map.index(index[0])
            j = compartment_map.index(index[1])
            array[i, j] = self.matrix[index]
        return np.array(array, copy=True)

    def full(self, parameter):
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

        size = len(self.compartment_map)
        mat = np.zeros((size, size), np.float64)
        for (to_comp, from_comp), param in self.matrix.items():
            to_idx = self.compartment_map.index(to_comp)
            fr_idx = self.compartment_map.index(from_comp)
            param = parameter.get(param)

            if to_idx == fr_idx:
                mat[to_idx, fr_idx] -= param
            else:
                mat[to_idx, fr_idx] += param
                mat[fr_idx, fr_idx] -= param
        return mat

    def eigen(self, parameter: ParameterGroup) -> Tuple[np.ndarray,
                                                        np.ndarray]:
        """ Returns the eigenvalues and eigenvectors of the k matrix.

        Parameters
        ----------
        parameter : glotaran.model.ParameterGroup


        Returns
        -------
        (eigenvalues, eigenvectos) : tuple(np.ndarray, np.ndarray)

        """
        matrix = self.full(parameter)
        # get the eigenvectors and values
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return (eigenvalues.real, eigenvectors.real)

    def a_matrix(self,
                 initial_concentration: InitialConcentration,
                 parameter: ParameterGroup) -> np.ndarray:
        initial_concentration = initial_concentration.parameter
        initial_concentration = \
            [initial_concentration[self._all_compartments.index(c)] for c in
             self.compartment_map]
        initial_concentration = [parameter.get(i) for i in initial_concentration]

        _, eigenvectors = self.eigen(parameter)
        gamma = np.matmul(scipy.linalg.inv(eigenvectors),
                          initial_concentration)

        a_matrix = np.empty(eigenvectors.shape, dtype=np.float64)

        for i in range(eigenvectors.shape[0]):
            a_matrix[i, :] = eigenvectors[:, i] * gamma[i]

        return a_matrix

    def __str__(self):
        """ """

        longest = max([len(s) for s in self.compartment_map]) + 6
        header = "compartment |"
        if longest < len(header):
            longest = len(header)

        longest_h = max([len(f" __{c}__ |") for c in self.compartment_map]) + 3
        longest_p = max([len(str(self.matrix[i])) for i in self.matrix]) + 3
        if longest_p < longest_h:
            longest_p = longest_h

        string = f"### _{self.label}_\n"
        string += "\n"
        #  string += "```\n"
        #  string += f"\t"
        header_sub = "|".rjust(longest, "-")
        for comp in self.compartment_map:
            header += f" __{comp}__ |".rjust(longest_p)
            header_sub += "|".rjust(longest_p, "-")
        string += header
        string += "\n"
        string += header_sub
        string += "\n"
        for comp in self.compartment_map:
            string += f"__{comp}__ |".rjust(longest)
            for comp2 in self.compartment_map:
                found = False
                for index in self.matrix:
                    if index[0] == comp and index[1] == comp2:
                        found = True
                        string += f" {self.matrix[index]} |".rjust(longest_p)
                if not found:
                    string += " |".rjust(longest_p)
            string += "\n"
        return string
