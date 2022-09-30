""" K-Matrix """
from __future__ import annotations

import itertools
from collections import OrderedDict

import numpy as np
from scipy.linalg import eig
from scipy.linalg import solve

from glotaran.model import ModelItem
from glotaran.model import ParameterType
from glotaran.model import item
from glotaran.utils.ipython import MarkdownStr


def calculate_gamma(eigenvectors: np.ndarray, initial_concentration: np.ndarray) -> np.ndarray:
    return np.diag(solve(eigenvectors, initial_concentration))


@item
class KMatrix(ModelItem):
    """A K-Matrix represents a first order differental system."""

    matrix: dict[tuple[str, str], ParameterType]

    @classmethod
    def empty(cls, label: str, compartments: list[str]) -> KMatrix:
        """Creates an empty K-Matrix. Useful for combining.

        Parameters
        ----------
        label :
            Label of the K-Matrix

        compartments :
            A list of all compartments in the model.
        """
        return cls(label, OrderedDict())

    def involved_compartments(self) -> list[str]:
        """A list of all compartments in the Matrix."""
        # TODO: find a better way that preserves ordering as defined in initial_concentrations
        compartments = []
        for index in self.matrix:
            if index[0] not in compartments:
                compartments.append(index[0])
            if index[1] not in compartments:
                compartments.append(index[1])

        # Don't use set, it randomly reorders the compartments.
        # compartments = list(set(compartments))
        return compartments

    def combine(self, k_matrix: KMatrix) -> KMatrix:
        """Creates a combined matrix.

        When combining k-matrices km1 and km2 (km1.combine(km2)),
        entries in km1 will be overwritten by corresponding entries in km2.

        Parameters
        ----------
        k_matrix :
            KMatrix to combine with.

        Returns
        -------
        combined :
            The combined KMatrix.

        """
        if not isinstance(k_matrix, KMatrix):
            raise TypeError("K-Matrices can only be combined with other K-Matrices.")
        combined_matrix = {entry: self.matrix[entry] for entry in self.matrix}
        for entry in k_matrix.matrix:
            combined_matrix[entry] = k_matrix.matrix[entry]
        return KMatrix(label=f"{self.label}+{k_matrix.label}", matrix=combined_matrix)

    def matrix_as_markdown(
        self,
        compartments: list[str] = None,
        fill_parameters: bool = False,
    ) -> MarkdownStr:
        """Returns the KMatrix as markdown formatted table.

        Parameters
        ----------
        compartments :
            (default = None)
            An optional list defining the desired order of compartments.
        fill_parameters : bool
            (default = False)
            If true, the entries will be filled with the actual parameter values
            instead of labels.
        """
        compartments = compartments or self.involved_compartments()
        size = len(compartments)
        array = np.zeros((size, size), dtype=object)
        # Matrix is a dict
        for index in self.matrix:
            i = compartments.index(index[0])
            j = compartments.index(index[1])
            array[i, j] = self.matrix[index].value if fill_parameters else self.matrix[index]

        return self._array_as_markdown(array, compartments, compartments)

    def _repr_markdown_(self) -> str:
        """Special method used by ``ipython`` to render markdown."""
        return str(self.matrix_as_markdown())

    def a_matrix_as_markdown(
        self, compartments: list[str], initial_concentration: np.ndarray
    ) -> MarkdownStr:
        """Returns the A Matrix as markdown formatted table.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        return self._array_as_markdown(
            self.a_matrix(compartments, initial_concentration).T,
            compartments,
            self.rates(compartments, initial_concentration),
        )

    @staticmethod
    def _array_as_markdown(array, row_header, column_header) -> MarkdownStr:
        markdown = "| compartment | " + " | ".join(
            e if isinstance(e, str) else f"{e:.4e}" for e in column_header
        )

        markdown += "\n|"
        markdown += "|".join("---" for _ in range(len(column_header) + 1))
        markdown += "\n"

        for i, row in enumerate(array):
            markdown += (
                f"| {row_header[i]} | "
                if isinstance(row_header[i], str)
                else f"| {row_header[i]:.4e} | "
            )
            markdown += " | ".join(e if isinstance(e, str) else f"{e:.4e}" for e in row)

            markdown += "|\n"

        return MarkdownStr(markdown)

    def reduced(self, compartments: list[str]) -> np.ndarray:
        """The reduced representation of the KMatrix as numpy array.

        Parameters
        ----------
        compartments :
            The compartment order.
        """

        size = len(compartments)
        array = np.zeros((size, size), dtype=np.float64)
        # Matrix is a dict
        for index in self.matrix:
            i = compartments.index(index[0])
            j = compartments.index(index[1])
            array[i, j] = self.matrix[index]
        return array

    def full(self, compartments: list[str]) -> np.ndarray:
        """The full representation of the KMatrix as numpy array.

        Parameters
        ----------
        compartments :
            The compartment order.
        """
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

    def eigen(self, compartments: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns the eigenvalues and eigenvectors of the k matrix.

        Parameters
        ----------
        compartments :
            The compartment order.
        """
        # We take the transpose to be consistent with timp
        matrix = self.full(compartments).T
        # get the eigenvectors and values, we take the left ones to have
        # computation consistent with TIMP
        eigenvalues, eigenvectors = eig(matrix, left=True, right=False)
        return (eigenvalues.real, eigenvectors.real)

    def rates(self, compartments: list[str], initial_concentration: np.ndarray) -> np.ndarray:
        """The resulting rates of the matrix.

        By definition, the eigenvalues of the compartmental model are negative and
        the rates are the negatives of the eigenvalues, thus the eigenvalues need to be
        multiplied with ``-1`` to get rates with the correct sign.

        Parameters
        ----------
        compartments: list[str]
            Names of compartment used to order the matrix.
        initial_concentration: np.ndarray
            The initial concentration.
        """
        if self.is_sequential(compartments, initial_concentration):
            return -np.diag(self.full(compartments)).copy()
        eigenvalues, _ = self.eigen(compartments)
        return -eigenvalues

    def a_matrix(self, compartments: list[str], initial_concentration: np.ndarray) -> np.ndarray:
        """The A matrix of the KMatrix.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        return (
            self.a_matrix_sequential(compartments)
            if self.is_sequential(compartments, initial_concentration)
            else self.a_matrix_general(compartments, initial_concentration)
        )

    def a_matrix_general(
        self, compartments: list[str], initial_concentration: np.ndarray
    ) -> np.ndarray:
        """The A matrix of the KMatrix for a general model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        _, eigenvectors = self.eigen(compartments)

        gamma = calculate_gamma(eigenvectors, initial_concentration)

        a_matrix = eigenvectors @ gamma

        return a_matrix.T

    def a_matrix_sequential(self, compartments: list[str]) -> np.ndarray:
        """The A matrix of the KMatrix for a sequential model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        matrix = self.full(compartments).T
        rates = np.diag(matrix)

        a_matrix = np.zeros(matrix.shape, dtype=np.float64)
        a_matrix[0, 0] = 1.0
        for i, j in itertools.product(range(rates.size), range(1, rates.size)):
            if i > j:
                continue
            a_matrix[i, j] = np.prod([rates[m] for m in range(j)]) / np.prod(
                [rates[m] - rates[i] for m in range(j + 1) if i != m]
            )

        return a_matrix

    def is_sequential(self, compartments: list[str], initial_concentration: np.ndarray) -> bool:
        """Returns true in the KMatrix represents an unibranched model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        if np.sum(initial_concentration) != 1:
            return False
        matrix = self.reduced(compartments)
        return not any(
            np.nonzero(matrix[:, i])[0].size != 1 or i != 0 and matrix[i, i - 1] == 0
            for i in range(matrix.shape[1])
        )
