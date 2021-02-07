""" K-Matrix """
from __future__ import annotations

import itertools
import typing
from collections import OrderedDict

import numpy as np
import scipy

from glotaran.model import model_attribute
from glotaran.parameter import Parameter

from .initial_concentration import InitialConcentration


@model_attribute(
    properties={
        "matrix": {"type": typing.Dict[typing.Tuple[str, str], Parameter]},
    },
)
class KMatrix:
    """ A K-Matrix represents a first order differental system."""

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
        """ A list of all compartments in the Matrix. """
        compartments = []
        for index in self.matrix:
            compartments.append(index[0])
            compartments.append(index[1])

        compartments = list(set(compartments))
        return compartments

    def combine(self, k_matrix: KMatrix) -> KMatrix:
        """Creates a combined matrix.

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
        combined_matrix = {entry: k_matrix.matrix[entry] for entry in k_matrix.matrix}
        for entry in self.matrix:
            combined_matrix[entry] = self.matrix[entry]
        combined = KMatrix()
        combined.label = f"{self.label}+{k_matrix.label}"
        combined.matrix = combined_matrix
        return combined

    def matrix_as_markdown(
        self,
        compartments: list[str] = None,
        fill_parameters: bool = False,
    ) -> str:
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

        compartments = (
            [c for c in compartments if c in self.involved_compartments()]
            if compartments
            else self.involved_compartments()
        )
        size = len(compartments)
        array = np.zeros((size, size), dtype=np.object)
        # Matrix is a dict
        for index in self.matrix:
            i = compartments.index(index[0])
            j = compartments.index(index[1])
            array[i, j] = (
                self.matrix[index].full_label if not fill_parameters else self.matrix[index].value
            )
        return self._array_as_markdown(array, compartments, compartments)

    def a_matrix_as_markdown(self, initial_concentration: InitialConcentration) -> str:
        """Returns the A Matrix as markdown formatted table.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        compartments = [
            c for c in initial_concentration.compartments if c in self.involved_compartments()
        ]
        return self._array_as_markdown(
            self.a_matrix(initial_concentration).T,
            compartments,
            self.rates(initial_concentration),
        )

    @staticmethod
    def _array_as_markdown(array, row_header, column_header):
        markdown = "| compartment | "
        markdown += " | ".join(f"{e:.4e}" if not isinstance(e, str) else e for e in column_header)

        markdown += "\n|"
        markdown += "|".join("---" for _ in range(len(column_header) + 1))
        markdown += "\n"

        for i, row in enumerate(array):
            markdown += (
                f"| {row_header[i]} | "
                if isinstance(row_header[i], str)
                else f"| {row_header[i]:.4e} | "
            )
            markdown += " | ".join(f"{e:.4e}" if not isinstance(e, str) else e for e in row)

            markdown += "|\n"

        return markdown

    def reduced(self, compartments: list[str]) -> np.ndarray:
        """The reduced representation of the KMatrix as numpy array.

        Parameters
        ----------
        compartments :
            The compartment order.
        """

        compartments = [c for c in compartments if c in self.involved_compartments()]
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
        eigenvalues, eigenvectors = scipy.linalg.eig(matrix, left=True, right=False)
        return (eigenvalues.real, eigenvectors.real)

    def rates(self, initial_concentration: InitialConcentration) -> np.ndarray:
        """The resulting rates of the matrix.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        if self.is_unibranched(initial_concentration):
            return np.diag(self.full(initial_concentration.compartments)).copy()
        rates, _ = self.eigen(initial_concentration.compartments)
        return rates

    def _gamma(
        self,
        eigenvectors: np.ndarray,
        initial_concentration: InitialConcentration,
    ) -> np.ndarray:
        compartments = [
            c for c in initial_concentration.compartments if c in self.involved_compartments()
        ]
        initial_concentration = [
            initial_concentration.parameters[initial_concentration.compartments.index(c)]
            for c in compartments
        ]

        gamma = scipy.linalg.solve(eigenvectors, initial_concentration)
        return np.diag(gamma)

    def a_matrix(self, initial_concentration: InitialConcentration) -> np.ndarray:
        """The resulting A matrix of the KMatrix.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        return (
            self.a_matrix_unibranch(initial_concentration)
            if self.is_unibranched(initial_concentration)
            else self.a_matrix_non_unibranch(initial_concentration)
        )

    def a_matrix_non_unibranch(self, initial_concentration: InitialConcentration) -> np.ndarray:
        """The resulting A matrix of the KMatrix for a non-unibranched model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        eigenvalues, eigenvectors = self.eigen(initial_concentration.compartments)
        gamma = self._gamma(eigenvectors, initial_concentration)

        a_matrix = eigenvectors @ gamma

        return a_matrix.T

    def a_matrix_unibranch(self, initial_concentration: InitialConcentration) -> np.array:
        """The resulting A matrix of the KMatrix for an unibranched model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        compartments = [
            c for c in initial_concentration.compartments if c in self.involved_compartments()
        ]
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

    def is_unibranched(self, initial_concentration: InitialConcentration) -> bool:
        """Returns true in the KMatrix represents an unibranched model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        if (
            np.sum(
                [
                    initial_concentration.parameters[initial_concentration.compartments.index(c)]
                    for c in self.involved_compartments()
                ]
            )
            != 1
        ):
            return False
        matrix = self.reduced(initial_concentration.compartments)
        return not any(
            np.nonzero(matrix[:, i])[0].size != 1 or i != 0 and matrix[i, i - 1] == 0
            for i in range(matrix.shape[1])
        )
